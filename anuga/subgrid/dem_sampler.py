"""
DEM sampling within mesh triangles and along mesh edges.

Uses GDAL (via ANUGA's spatialInputUtil) to sample fine-resolution
raster elevations inside each triangular cell, then builds piecewise-
linear V(eta) and A_w(eta) lookup tables.
"""

import numpy as np
from anuga.subgrid.subgrid_tables import CellVolumeTable, EdgeAreaTable


def _triangle_vertices(domain, cell_id):
    """Return the 3 vertices of triangle cell_id as a (3,2) array in absolute coordinates.

    Uses absolute coordinates (adds mesh geo_reference offset if needed) so that
    the returned vertices can be used to query a georeferenced raster DEM.
    """
    i0 = 3 * cell_id
    v = domain.vertex_coordinates[i0:i0 + 3, :].copy()
    gr = domain.geo_reference
    if not gr.is_absolute():
        v[:, 0] += gr.get_xllcorner()
        v[:, 1] += gr.get_yllcorner()
    return v


def _edge_endpoints(domain, cell_id, edge_index):
    """Return the two endpoints of edge `edge_index` of triangle `cell_id`.

    ANUGA convention: edge i is opposite vertex i.
      edge 0 = vertices 1-2
      edge 1 = vertices 2-0
      edge 2 = vertices 0-1

    Returns
    -------
    ndarray, shape (2, 2)
        Two (x, y) endpoints of the edge.
    """
    verts = _triangle_vertices(domain, cell_id)
    edge_vertex_map = [(1, 2), (2, 0), (0, 1)]
    v0, v1 = edge_vertex_map[edge_index]
    return np.array([verts[v0], verts[v1]])


def _sample_dem_in_triangle(triangle_verts, dem_file, sampling_resolution):
    """Sample DEM elevations on a regular grid inside a triangle.

    Parameters
    ----------
    triangle_verts : ndarray, shape (3, 2)
        Vertices of the triangle.
    dem_file : str
        Path to GDAL-compatible raster DEM.
    sampling_resolution : float
        Approximate grid spacing in map units (metres).

    Returns
    -------
    elevations : ndarray
        1D array of DEM elevations inside the triangle (nodata removed).
    pixel_area : float
        Area represented by each sample point.
    """
    from anuga.utilities.spatialInputUtil import gridPointsInPolygon, rasterValuesAtPoints

    polygon = triangle_verts.tolist()

    grid_spacing = [sampling_resolution, sampling_resolution]
    try:
        xy = gridPointsInPolygon(polygon, approx_grid_spacing=grid_spacing)
    except Exception:
        # Degenerate triangle or too small for even 4x4 grid
        return np.array([]), 0.0

    if len(xy) == 0:
        return np.array([]), 0.0

    elevations = rasterValuesAtPoints(xy, dem_file, interpolation='pixel')

    # Compute pixel area BEFORE removing nodata, so each sample point
    # represents an equal share of the triangle regardless of nodata gaps.
    n_total = len(elevations)
    from anuga.geometry.polygon import polygon_area as _polygon_area
    tri_area = abs(_polygon_area(triangle_verts))
    pixel_area = tri_area / n_total

    # Remove nodata (NaN)
    valid = np.isfinite(elevations)
    elevations = elevations[valid]

    if len(elevations) == 0:
        return np.array([]), 0.0

    return elevations, pixel_area


def _build_volume_table(elevations, pixel_area, n_breakpoints, cell_area):
    """Build piecewise-linear V(eta) and A_w(eta) from sampled elevations.

    Algorithm:
      1. Sort elevations ascending.
      2. At each elevation z_i, the cumulative volume when stage = z_i is
         V(z_i) = sum_{j < i} (z_i - z_j) * pixel_area
      3. Wet area A_w(z_i) = i * pixel_area  (number of submerged pixels
         times their area).
      4. Compress to n_breakpoints using uniform spacing in elevation.

    Parameters
    ----------
    elevations : ndarray
        DEM elevations inside the cell (will be sorted internally).
    pixel_area : float
        Area per sample point.
    n_breakpoints : int
        Number of breakpoints in the compressed table.
    cell_area : float
        Total cell area from mesh geometry.

    Returns
    -------
    eta : ndarray
        Elevation breakpoints (length <= n_breakpoints).
    volume : ndarray
        Cumulative wet volume at each breakpoint.
    wet_area : ndarray
        Wet planform area at each breakpoint.
    """
    if len(elevations) == 0:
        return np.array([]), np.array([]), np.array([])

    z = np.sort(elevations)
    n_pts = len(z)

    z_min = z[0]
    z_max = z[-1]

    if z_min == z_max:
        # Flat bed: trivial table with 2 breakpoints
        return (
            np.array([z_min, z_min + 1.0]),
            np.array([0.0, cell_area * 1.0]),
            np.array([cell_area, cell_area]),
        )

    # Build full-resolution cumulative volume and wet area
    # V(eta) at each sorted elevation: V(z_i) = pixel_area * sum_{j<=i} (z_i - z_j)
    # This is equivalent to: V(z_i) = pixel_area * [(i+1)*z_i - cumsum_z[i]]
    cumsum_z = np.cumsum(z)
    indices = np.arange(1, n_pts + 1, dtype=np.float64)
    V_full = pixel_area * (indices * z - cumsum_z)
    A_full = indices * pixel_area

    # Choose breakpoint elevations: uniform spacing from z_min to z_max
    n_bp = min(n_breakpoints, n_pts)
    if n_bp < 2:
        n_bp = 2

    eta_bp = np.linspace(z_min, z_max, n_bp)

    # Interpolate V and A at breakpoints
    # Use the full-resolution z as x-axis (may have duplicates, interp handles this)
    V_bp = np.interp(eta_bp, z, V_full)
    A_bp = np.interp(eta_bp, z, A_full)

    # Ensure V(z_min) = 0 and A(z_min) = 0 (nothing wet at lowest point)
    V_bp[0] = 0.0
    A_bp[0] = 0.0

    # Cap wet area at cell area
    A_bp = np.minimum(A_bp, cell_area)

    return eta_bp, V_bp, A_bp


def sample_dem_for_cells(domain, dem_file, sampling_resolution=2.0,
                         n_breakpoints=20, verbose=False):
    """Sample a DEM within every mesh triangle and build cell volume tables.

    Parameters
    ----------
    domain : anuga.Domain
        The ANUGA domain (provides mesh geometry).
    dem_file : str
        Path to a GDAL-compatible raster DEM.
    sampling_resolution : float
        Approximate spacing of sample points within each triangle (metres).
    n_breakpoints : int
        Maximum number of elevation breakpoints per cell.
    verbose : bool
        Print progress if True.

    Returns
    -------
    CellVolumeTable
        Populated lookup tables for all cells.
    """
    n_cells = domain.number_of_elements
    table = CellVolumeTable(n_cells, max_breaks=n_breakpoints)

    areas = domain.areas

    for k in range(n_cells):
        if verbose and k % 1000 == 0:
            print(f"  Sampling cell {k}/{n_cells}...")

        verts = _triangle_vertices(domain, k)
        cell_area = float(areas[k])

        elevations, pixel_area = _sample_dem_in_triangle(
            verts, dem_file, sampling_resolution
        )

        if len(elevations) == 0:
            # Fallback: use mesh vertex elevations for a flat-bed table
            elev_v = domain.quantities['elevation'].vertex_values[k, :]
            z_avg = float(np.mean(elev_v))
            table.set_cell(k,
                           np.array([z_avg, z_avg + 1.0]),
                           np.array([0.0, cell_area]),
                           np.array([cell_area, cell_area]),
                           cell_area)
            continue

        eta, volume, wet_area = _build_volume_table(
            elevations, pixel_area, n_breakpoints, cell_area
        )

        if len(eta) >= 2:
            table.set_cell(k, eta, volume, wet_area, cell_area)
        else:
            # Fallback
            z_avg = float(np.mean(elevations))
            table.set_cell(k,
                           np.array([z_avg, z_avg + 1.0]),
                           np.array([0.0, cell_area]),
                           np.array([cell_area, cell_area]),
                           cell_area)

    if verbose:
        print(f"  Cell volume tables complete for {n_cells} cells.")

    return table


def _sample_dem_along_edge(edge_endpoints, dem_file, sampling_resolution):
    """Sample DEM elevations along an edge at regular intervals.

    Parameters
    ----------
    edge_endpoints : ndarray, shape (2, 2)
        Start and end points of the edge.
    dem_file : str
        Path to GDAL-compatible raster DEM.
    sampling_resolution : float
        Approximate spacing of sample points along the edge.

    Returns
    -------
    elevations : ndarray
        1D array of DEM elevations along the edge.
    edge_length : float
        Total length of the edge.
    sample_spacing : float
        Actual spacing between sample points.
    """
    from anuga.utilities.spatialInputUtil import rasterValuesAtPoints

    p0 = edge_endpoints[0]
    p1 = edge_endpoints[1]
    edge_length = float(np.linalg.norm(p1 - p0))

    if edge_length < 1e-12:
        return np.array([]), 0.0, 0.0

    n_samples = max(int(np.ceil(edge_length / sampling_resolution)), 2)
    t = np.linspace(0, 1, n_samples)
    xy = np.outer(1 - t, p0) + np.outer(t, p1)

    elevations = rasterValuesAtPoints(xy, dem_file, interpolation='pixel')
    valid = np.isfinite(elevations)
    elevations = elevations[valid]

    sample_spacing = edge_length / (n_samples - 1) if n_samples > 1 else edge_length

    return elevations, edge_length, sample_spacing


def _build_edge_area_table(elevations, edge_length, sample_spacing, n_breakpoints):
    """Build piecewise-linear A_f(eta) and B_f(eta) for an edge cross-section.

    The cross-section is approximated as a sequence of vertical columns
    of width = sample_spacing, each with bed elevation z_i.

    A_f(eta) = sum of max(0, eta - z_i) * sample_spacing  (flow area)
    B_f(eta) = number of submerged columns * sample_spacing  (flow width)

    Parameters
    ----------
    elevations : ndarray
        DEM elevations along the edge.
    edge_length : float
        Total edge length.
    sample_spacing : float
        Distance between sample points.
    n_breakpoints : int
        Number of elevation breakpoints.

    Returns
    -------
    eta : ndarray
        Elevation breakpoints.
    flow_area : ndarray
        Cross-sectional flow area at each breakpoint.
    flow_width : ndarray
        Flow width at each breakpoint.
    """
    if len(elevations) == 0:
        return np.array([]), np.array([]), np.array([])

    z = np.sort(elevations)
    n_pts = len(z)
    z_min = z[0]
    z_max = z[-1]

    if z_min == z_max:
        return (
            np.array([z_min, z_min + 1.0]),
            np.array([0.0, edge_length * 1.0]),
            np.array([edge_length, edge_length]),
        )

    # Full-resolution tables
    cumsum_z = np.cumsum(z)
    indices = np.arange(1, n_pts + 1, dtype=np.float64)
    A_full = sample_spacing * (indices * z - cumsum_z)
    W_full = indices * sample_spacing

    n_bp = min(n_breakpoints, n_pts)
    if n_bp < 2:
        n_bp = 2

    eta_bp = np.linspace(z_min, z_max, n_bp)
    A_bp = np.interp(eta_bp, z, A_full)
    W_bp = np.interp(eta_bp, z, W_full)

    A_bp[0] = 0.0
    W_bp[0] = 0.0
    W_bp = np.minimum(W_bp, edge_length)

    return eta_bp, A_bp, W_bp


def sample_dem_for_edges(domain, dem_file, sampling_resolution=2.0,
                         n_breakpoints=20, verbose=False):
    """Sample a DEM along every mesh edge and build edge area tables.

    This is the stretch-goal edge table construction.

    Parameters
    ----------
    domain : anuga.Domain
        The ANUGA domain.
    dem_file : str
        Path to a GDAL-compatible raster DEM.
    sampling_resolution : float
        Approximate spacing along edges (metres).
    n_breakpoints : int
        Maximum breakpoints per edge.
    verbose : bool
        Print progress if True.

    Returns
    -------
    EdgeAreaTable
        Populated cross-section tables for all edges.
    """
    n_cells = domain.number_of_elements
    n_edges = 3 * n_cells
    table = EdgeAreaTable(n_edges, max_breaks=n_breakpoints)

    for k in range(n_cells):
        if verbose and k % 1000 == 0:
            print(f"  Sampling edges for cell {k}/{n_cells}...")

        for i in range(3):
            edge_id = 3 * k + i
            endpoints = _edge_endpoints(domain, k, i)

            elevations, edge_length, sample_spacing = _sample_dem_along_edge(
                endpoints, dem_file, sampling_resolution
            )

            if len(elevations) < 2:
                # Fallback: single elevation
                bed_edge = domain.quantities['elevation'].edge_values[k, i]
                table.set_edge(edge_id,
                               np.array([bed_edge, bed_edge + 1.0]),
                               np.array([0.0, edge_length]),
                               np.array([edge_length, edge_length]))
                continue

            eta, flow_area, flow_width = _build_edge_area_table(
                elevations, edge_length, sample_spacing, n_breakpoints
            )

            if len(eta) >= 2:
                table.set_edge(edge_id, eta, flow_area, flow_width)
            else:
                bed_edge = domain.quantities['elevation'].edge_values[k, i]
                table.set_edge(edge_id,
                               np.array([bed_edge, bed_edge + 1.0]),
                               np.array([0.0, edge_length]),
                               np.array([edge_length, edge_length]))

    if verbose:
        print(f"  Edge area tables complete for {n_edges} edges.")

    return table
