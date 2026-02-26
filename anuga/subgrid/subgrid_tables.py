"""
Sub-grid lookup tables for volume-elevation and flow-area-elevation
relationships within mesh cells and along mesh edges.

Each cell stores a piecewise-linear V(eta) curve:
    V(eta) = cumulative wet volume at stage eta
    A_w(eta) = wet planform area at stage eta

Each edge stores (stretch goal):
    A_f(eta) = cross-sectional flow area at stage eta
    B_f(eta) = flow width at stage eta
"""

import numpy as np


class CellVolumeTable:
    """Per-cell volume-elevation lookup tables.

    For each triangular cell, stores a piecewise-linear relationship
    between water surface elevation (eta) and:
      - Cumulative wet volume V(eta)
      - Wet planform area A_w(eta)

    The tables are built from fine-resolution DEM sampling within each
    triangle.  At runtime, V(eta) is evaluated by linear interpolation
    between breakpoints, and eta = V^{-1}(V) is evaluated by binary
    search on the monotonically increasing V column.

    Parameters
    ----------
    n_cells : int
        Number of triangular cells in the mesh.
    max_breaks : int
        Maximum number of elevation breakpoints per cell (default 20).
        Cells with fewer valid breakpoints are padded with NaN.
    """

    def __init__(self, n_cells, max_breaks=20):
        self.n_cells = n_cells
        self.max_breaks = max_breaks

        # Per-cell piecewise-linear tables (padded with NaN)
        self.eta_breaks = np.full((n_cells, max_breaks), np.nan, dtype=np.float64)
        self.vol_cumul = np.full((n_cells, max_breaks), np.nan, dtype=np.float64)
        self.wet_area = np.full((n_cells, max_breaks), np.nan, dtype=np.float64)

        # Number of valid breakpoints per cell
        self.n_breaks = np.zeros(n_cells, dtype=np.int32)

        # Min/max DEM elevation within each cell
        self.z_min = np.full(n_cells, np.nan, dtype=np.float64)
        self.z_max = np.full(n_cells, np.nan, dtype=np.float64)

        # Cell areas (from mesh, for consistency checks)
        self.cell_areas = np.zeros(n_cells, dtype=np.float64)

    def set_cell(self, cell_id, eta, volume, area, cell_area):
        """Set the lookup table for a single cell.

        Parameters
        ----------
        cell_id : int
            Index of the cell.
        eta : array_like
            Sorted elevation breakpoints (ascending).
        volume : array_like
            Cumulative wet volume at each breakpoint.
        area : array_like
            Wet planform area at each breakpoint.
        cell_area : float
            Total planform area of the cell (from mesh geometry).
        """
        eta = np.asarray(eta, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        area = np.asarray(area, dtype=np.float64)

        n = len(eta)
        if len(volume) != n or len(area) != n:
            raise ValueError(
                f"Cell {cell_id}: eta, volume, area must have the same length. "
                f"Got len(eta)={n}, len(volume)={len(volume)}, len(area)={len(area)}."
            )
        if n > self.max_breaks:
            raise ValueError(
                f"Cell {cell_id}: {n} breakpoints exceeds max_breaks={self.max_breaks}. "
                "Increase max_breaks or compress the table."
            )
        if n >= 2:
            if np.any(np.diff(eta) < 0):
                raise ValueError(
                    f"Cell {cell_id}: eta breakpoints must be non-decreasing."
                )
            if np.any(np.diff(volume) < 0):
                raise ValueError(
                    f"Cell {cell_id}: volume must be non-decreasing."
                )

        self.n_breaks[cell_id] = n
        self.eta_breaks[cell_id, :n] = eta
        self.vol_cumul[cell_id, :n] = volume
        self.wet_area[cell_id, :n] = area
        self.z_min[cell_id] = eta[0]
        self.z_max[cell_id] = eta[-1]
        self.cell_areas[cell_id] = cell_area

    def volume_from_stage(self, eta, cell_id):
        """Evaluate V(eta) for a single cell via piecewise-linear interpolation.

        Parameters
        ----------
        eta : float
            Water surface elevation.
        cell_id : int
            Cell index.

        Returns
        -------
        float
            Wet volume at stage eta.
        """
        n = self.n_breaks[cell_id]
        if n == 0:
            return 0.0

        breaks = self.eta_breaks[cell_id, :n]
        vols = self.vol_cumul[cell_id, :n]

        if eta <= breaks[0]:
            return 0.0
        if eta >= breaks[-1]:
            # Linear extrapolation above table: full cell area
            return vols[-1] + self.cell_areas[cell_id] * (eta - breaks[-1])

        # Binary search for interval
        idx = np.searchsorted(breaks, eta, side='right') - 1
        # Linear interpolation
        t = (eta - breaks[idx]) / (breaks[idx + 1] - breaks[idx])
        return vols[idx] + t * (vols[idx + 1] - vols[idx])

    def stage_from_volume(self, V, cell_id):
        """Invert V(eta) -> eta for a single cell via binary search.

        Parameters
        ----------
        V : float
            Target wet volume.
        cell_id : int
            Cell index.

        Returns
        -------
        float
            Water surface elevation corresponding to volume V.
        """
        n = self.n_breaks[cell_id]
        if n == 0 or V <= 0.0:
            return self.z_min[cell_id] if n > 0 else 0.0

        vols = self.vol_cumul[cell_id, :n]

        if V >= vols[-1]:
            # Above table range: linear extrapolation
            area = self.cell_areas[cell_id]
            if area > 0:
                return self.eta_breaks[cell_id, n - 1] + (V - vols[-1]) / area
            else:
                return self.eta_breaks[cell_id, n - 1]

        # Binary search for volume interval
        idx = np.searchsorted(vols, V, side='right') - 1
        idx = max(idx, 0)

        breaks = self.eta_breaks[cell_id, :n]
        dV = vols[idx + 1] - vols[idx]
        if dV > 0:
            t = (V - vols[idx]) / dV
        else:
            t = 0.0
        return breaks[idx] + t * (breaks[idx + 1] - breaks[idx])

    def wet_area_from_stage(self, eta, cell_id):
        """Evaluate A_w(eta) for a single cell.

        Parameters
        ----------
        eta : float
            Water surface elevation.
        cell_id : int
            Cell index.

        Returns
        -------
        float
            Wet planform area at stage eta.
        """
        n = self.n_breaks[cell_id]
        if n == 0:
            return 0.0

        breaks = self.eta_breaks[cell_id, :n]
        areas = self.wet_area[cell_id, :n]

        if eta <= breaks[0]:
            return 0.0
        if eta >= breaks[-1]:
            return self.cell_areas[cell_id]

        idx = np.searchsorted(breaks, eta, side='right') - 1
        t = (eta - breaks[idx]) / (breaks[idx + 1] - breaks[idx])
        return areas[idx] + t * (areas[idx + 1] - areas[idx])

    # --- Vectorized operations for the full mesh ---

    def volume_from_stage_vec(self, eta_array):
        """Evaluate V(eta) for all cells at once.

        Parameters
        ----------
        eta_array : ndarray, shape (n_cells,)
            Water surface elevation at each cell centroid.

        Returns
        -------
        ndarray, shape (n_cells,)
            Wet volume for each cell.
        """
        volumes = np.zeros(self.n_cells, dtype=np.float64)
        for k in range(self.n_cells):
            volumes[k] = self.volume_from_stage(eta_array[k], k)
        return volumes

    def stage_from_volume_vec(self, V_array):
        """Invert V(eta) -> eta for all cells at once.

        Parameters
        ----------
        V_array : ndarray, shape (n_cells,)
            Wet volume at each cell.

        Returns
        -------
        ndarray, shape (n_cells,)
            Water surface elevation for each cell.
        """
        stages = np.zeros(self.n_cells, dtype=np.float64)
        for k in range(self.n_cells):
            stages[k] = self.stage_from_volume(V_array[k], k)
        return stages

    def wet_area_from_stage_vec(self, eta_array):
        """Evaluate A_w(eta) for all cells at once.

        Parameters
        ----------
        eta_array : ndarray, shape (n_cells,)
            Water surface elevation at each cell centroid.

        Returns
        -------
        ndarray, shape (n_cells,)
            Wet planform area for each cell.
        """
        areas = np.zeros(self.n_cells, dtype=np.float64)
        for k in range(self.n_cells):
            areas[k] = self.wet_area_from_stage(eta_array[k], k)
        return areas


class EdgeAreaTable:
    """Per-edge cross-sectional flow area lookup tables (stretch goal).

    For each edge in the mesh, stores:
      - A_f(eta) = cross-sectional flow area
      - B_f(eta) = flow width

    Parameters
    ----------
    n_edges : int
        Number of edges (= 3 * n_cells for an internal edge-based layout).
    max_breaks : int
        Maximum number of elevation breakpoints per edge.
    """

    def __init__(self, n_edges, max_breaks=20):
        self.n_edges = n_edges
        self.max_breaks = max_breaks

        self.eta_breaks = np.full((n_edges, max_breaks), np.nan, dtype=np.float64)
        self.flow_area = np.full((n_edges, max_breaks), np.nan, dtype=np.float64)
        self.flow_width = np.full((n_edges, max_breaks), np.nan, dtype=np.float64)
        self.n_breaks = np.zeros(n_edges, dtype=np.int32)

    def set_edge(self, edge_id, eta, area, width):
        """Set the lookup table for a single edge.

        Parameters
        ----------
        edge_id : int
            Index of the edge (0 to 3*n_cells-1).
        eta : array_like
            Sorted elevation breakpoints (ascending).
        area : array_like
            Cross-sectional flow area at each breakpoint.
        width : array_like
            Flow width at each breakpoint.
        """
        eta = np.asarray(eta, dtype=np.float64)
        area = np.asarray(area, dtype=np.float64)
        width = np.asarray(width, dtype=np.float64)

        n = len(eta)
        if len(area) != n or len(width) != n:
            raise ValueError(
                f"Edge {edge_id}: eta, area, width must have the same length. "
                f"Got len(eta)={n}, len(area)={len(area)}, len(width)={len(width)}."
            )
        if n > self.max_breaks:
            raise ValueError(
                f"Edge {edge_id}: {n} breakpoints exceeds max_breaks={self.max_breaks}."
            )

        self.n_breaks[edge_id] = n
        self.eta_breaks[edge_id, :n] = eta
        self.flow_area[edge_id, :n] = area
        self.flow_width[edge_id, :n] = width

    def flow_area_from_stage(self, eta, edge_id):
        """Evaluate A_f(eta) for a single edge.

        Parameters
        ----------
        eta : float
            Water surface elevation.
        edge_id : int
            Edge index.

        Returns
        -------
        float
            Cross-sectional flow area at stage eta.
        """
        n = self.n_breaks[edge_id]
        if n == 0:
            return 0.0

        breaks = self.eta_breaks[edge_id, :n]
        areas = self.flow_area[edge_id, :n]

        if eta <= breaks[0]:
            return 0.0
        if eta >= breaks[-1]:
            # Extrapolate: last width * excess stage
            last_width = self.flow_width[edge_id, n - 1]
            return areas[-1] + last_width * (eta - breaks[-1])

        idx = np.searchsorted(breaks, eta, side='right') - 1
        t = (eta - breaks[idx]) / (breaks[idx + 1] - breaks[idx])
        return areas[idx] + t * (areas[idx + 1] - areas[idx])

    def flow_width_from_stage(self, eta, edge_id):
        """Evaluate B_f(eta) for a single edge.

        Parameters
        ----------
        eta : float
            Water surface elevation.
        edge_id : int
            Edge index.

        Returns
        -------
        float
            Flow width at stage eta.
        """
        n = self.n_breaks[edge_id]
        if n == 0:
            return 0.0

        breaks = self.eta_breaks[edge_id, :n]
        widths = self.flow_width[edge_id, :n]

        if eta <= breaks[0]:
            return 0.0
        if eta >= breaks[-1]:
            return widths[-1]

        idx = np.searchsorted(breaks, eta, side='right') - 1
        t = (eta - breaks[idx]) / (breaks[idx + 1] - breaks[idx])
        return widths[idx] + t * (widths[idx + 1] - widths[idx])
