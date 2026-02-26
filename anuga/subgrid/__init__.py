"""
Sub-grid terrain sampling for ANUGA.

Provides volume-elevation and flow-area-elevation lookup tables that let
coarse mesh elements capture fine-resolution terrain features from a DEM.

Based on:
  - Casulli (2009) for structured grids
  - Verschuren et al. (2026) for unstructured triangular meshes

Typical usage::

    domain = anuga.Domain(points, vertices, boundary)
    domain.set_subgrid_dem('/path/to/dem.tif',
                           sampling_resolution=2.0,
                           n_breakpoints=20,
                           cache=True)
"""

from anuga.subgrid.subgrid_tables import CellVolumeTable, EdgeAreaTable
from anuga.subgrid.dem_sampler import sample_dem_for_cells, sample_dem_for_edges
from anuga.subgrid.subgrid_io import save_subgrid_tables, load_subgrid_tables
from anuga.subgrid.operators import (SubGridData, SubGridCorrectionOperator,
                                     SubGridVolumeCorrector)
