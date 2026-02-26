"""
Serialization of sub-grid lookup tables to compressed .npz files.

Cache files are keyed by a hash of the mesh geometry, DEM metadata,
and sampling parameters so that tables are recomputed only when inputs
change.
"""

import hashlib
import os
import numpy as np
from anuga.subgrid.subgrid_tables import CellVolumeTable, EdgeAreaTable


def _compute_cache_key(domain, dem_file, sampling_resolution, n_breakpoints):
    """Compute a deterministic hash from mesh geometry + DEM + params.

    The hash incorporates:
      - Vertex coordinates (defines the mesh)
      - DEM file path and modification time
      - Sampling parameters

    Parameters
    ----------
    domain : anuga.Domain
        The ANUGA domain.
    dem_file : str
        Path to the DEM raster.
    sampling_resolution : float
        Sub-grid sampling resolution.
    n_breakpoints : int
        Number of table breakpoints.

    Returns
    -------
    str
        Hex digest of the cache key.
    """
    h = hashlib.sha256()

    # Mesh geometry
    v = domain.vertex_coordinates
    h.update(v.tobytes())

    # DEM file identity
    h.update(dem_file.encode('utf-8'))
    if os.path.isfile(dem_file):
        stat = os.stat(dem_file)
        h.update(str(stat.st_size).encode('utf-8'))
        h.update(str(stat.st_mtime_ns).encode('utf-8'))

    # Sampling parameters
    h.update(str(sampling_resolution).encode('utf-8'))
    h.update(str(n_breakpoints).encode('utf-8'))

    return h.hexdigest()[:16]


def save_subgrid_tables(filepath, cell_table, edge_table=None):
    """Save sub-grid tables to a compressed .npz file.

    Parameters
    ----------
    filepath : str
        Output file path (should end with .npz or .sgrid).
    cell_table : CellVolumeTable
        Cell volume-elevation tables.
    edge_table : EdgeAreaTable, optional
        Edge cross-section tables (stretch goal).
    """
    data = {
        'cell_n_cells': np.array([cell_table.n_cells]),
        'cell_max_breaks': np.array([cell_table.max_breaks]),
        'cell_eta_breaks': cell_table.eta_breaks,
        'cell_vol_cumul': cell_table.vol_cumul,
        'cell_wet_area': cell_table.wet_area,
        'cell_n_breaks': cell_table.n_breaks,
        'cell_z_min': cell_table.z_min,
        'cell_z_max': cell_table.z_max,
        'cell_areas': cell_table.cell_areas,
    }

    if edge_table is not None:
        data.update({
            'edge_n_edges': np.array([edge_table.n_edges]),
            'edge_max_breaks': np.array([edge_table.max_breaks]),
            'edge_eta_breaks': edge_table.eta_breaks,
            'edge_flow_area': edge_table.flow_area,
            'edge_flow_width': edge_table.flow_width,
            'edge_n_breaks': edge_table.n_breaks,
        })

    np.savez_compressed(filepath, **data)


def load_subgrid_tables(filepath):
    """Load sub-grid tables from a .npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.

    Returns
    -------
    cell_table : CellVolumeTable
        Loaded cell volume tables.
    edge_table : EdgeAreaTable or None
        Loaded edge tables, or None if not present.
    """
    # Handle .npz extension
    if not filepath.endswith('.npz'):
        filepath_load = filepath + '.npz'
    else:
        filepath_load = filepath

    data = np.load(filepath_load, allow_pickle=False)

    n_cells = int(data['cell_n_cells'][0])
    max_breaks = int(data['cell_max_breaks'][0])

    cell_table = CellVolumeTable(n_cells, max_breaks=max_breaks)
    cell_table.eta_breaks = data['cell_eta_breaks']
    cell_table.vol_cumul = data['cell_vol_cumul']
    cell_table.wet_area = data['cell_wet_area']
    cell_table.n_breaks = data['cell_n_breaks']
    cell_table.z_min = data['cell_z_min']
    cell_table.z_max = data['cell_z_max']
    cell_table.cell_areas = data['cell_areas']

    edge_table = None
    if 'edge_n_edges' in data:
        n_edges = int(data['edge_n_edges'][0])
        edge_max = int(data['edge_max_breaks'][0])
        edge_table = EdgeAreaTable(n_edges, max_breaks=edge_max)
        edge_table.eta_breaks = data['edge_eta_breaks']
        edge_table.flow_area = data['edge_flow_area']
        edge_table.flow_width = data['edge_flow_width']
        edge_table.n_breaks = data['edge_n_breaks']

    return cell_table, edge_table


def get_cache_filepath(domain, dem_file, sampling_resolution, n_breakpoints,
                       cache_dir=None):
    """Get the cache file path for a given mesh+DEM+params combination.

    Parameters
    ----------
    domain : anuga.Domain
        The ANUGA domain.
    dem_file : str
        Path to the DEM raster.
    sampling_resolution : float
        Sub-grid sampling resolution.
    n_breakpoints : int
        Number of table breakpoints.
    cache_dir : str, optional
        Directory for cache files. Defaults to same directory as dem_file.

    Returns
    -------
    str
        Full path for the cache file (without .npz extension).
    """
    key = _compute_cache_key(domain, dem_file, sampling_resolution, n_breakpoints)

    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(dem_file))

    return os.path.join(cache_dir, f'subgrid_cache_{key}')
