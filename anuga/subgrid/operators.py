"""
Sub-grid correction operator for ANUGA.

The SubGridCorrectionOperator is a fractional step operator that applies
volume-conservation corrections using pre-computed sub-grid terrain tables.

Without sub-grid: ANUGA conserves stage (w = z + h) directly, assuming
each cell has a single flat bed elevation.

With sub-grid: the actual volume-elevation relationship V(eta) is
non-linear due to terrain variability within the cell. This operator
corrects the standard flat-bed update to conserve volume properly.

After the standard flux update computes:
    stage_new = stage_old + dt * explicit_update

This operator:
    1. Converts the OLD stage to volume:  V_old = V(stage_old)
    2. Computes the volume change:        dV = (stage_new - stage_old) * cell_area
    3. Updates volume:                    V_new = V_old + dV
    4. Inverts to get corrected stage:    stage_corrected = V^{-1}(V_new)

The net effect is that the same flux divergence, when interpreted as a
volume change rather than a stage change, produces the physically correct
water surface elevation given the sub-grid terrain.
"""

import numpy as np
from anuga.operators.base_operator import Operator
from anuga.subgrid.subgrid_tables import CellVolumeTable


class SubGridData:
    """Container for sub-grid data attached to a Domain.

    Follows the RiverWall pattern: stores tables and state that the
    solver and operators can access via ``domain.subgridData``.

    Parameters
    ----------
    domain : anuga.Domain
        The parent domain.
    """

    def __init__(self, domain):
        self.domain = domain
        self.enabled = False

        # Lookup tables (set by set_subgrid_dem)
        self.cell_table = None
        self.edge_table = None

        # Runtime state arrays (allocated when tables are set)
        self.volume_centroid = None      # V at centroid for each cell
        self.wet_area_centroid = None    # A_w at centroid for each cell

    def set_tables(self, cell_table, edge_table=None):
        """Attach pre-computed sub-grid tables.

        Parameters
        ----------
        cell_table : CellVolumeTable
            Per-cell volume-elevation tables.
        edge_table : EdgeAreaTable, optional
            Per-edge flow area tables (stretch goal).
        """
        n = self.domain.number_of_elements
        if cell_table.n_cells != n:
            raise ValueError(
                f"Cell table has {cell_table.n_cells} cells, "
                f"domain has {n} elements."
            )

        self.cell_table = cell_table
        self.edge_table = edge_table
        self.enabled = True

        # Allocate runtime state
        self.volume_centroid = np.zeros(n, dtype=np.float64)
        self.wet_area_centroid = np.zeros(n, dtype=np.float64)


class SubGridCorrectionOperator(Operator):
    """Fractional step operator that corrects stage updates for sub-grid terrain.

    This operator runs after the standard conserved quantity update.
    It converts the flat-bed stage update into a volume-conserving update
    using the pre-computed V(eta) tables.

    Parameters
    ----------
    domain : anuga.Domain
        The domain, which must have ``domain.subgridData`` with tables set.
    description : str, optional
        Operator description.
    label : str, optional
        Operator label.
    """

    def __init__(self, domain, description=None, label=None,
                 logging=False, verbose=False):

        if description is None:
            description = 'Sub-grid volume-conservation correction'
        if label is None:
            label = 'subgrid_correction'

        Operator.__init__(self, domain, description, label, logging, verbose)

        if not hasattr(domain, 'subgridData') or not domain.subgridData.enabled:
            raise RuntimeError(
                "SubGridCorrectionOperator requires domain.subgridData "
                "to be set up first. Call domain.set_subgrid_dem() before "
                "creating this operator."
            )

        self.sg = domain.subgridData
        self.table = self.sg.cell_table

        # Track cumulative absolute volume correction for diagnostics
        self._total_volume_correction = 0.0

    def __call__(self):
        """Apply sub-grid volume correction.

        The standard update has already modified stage_centroid_values.
        We recompute effective height from the sub-grid V(eta)/A_w(eta)
        tables and kill momentum in dry cells.
        """
        stage_c = self.stage_c
        areas = self.areas
        n = self.domain.number_of_elements
        table = self.table
        sg = self.sg

        # For each cell, compute the effective height and correct
        total_correction = 0.0
        for k in range(n):
            eta = stage_c[k]

            # Compute volume from sub-grid table
            V = table.volume_from_stage(eta, k)

            # Compute wet area from sub-grid table
            A_w = table.wet_area_from_stage(eta, k)

            # Store for use by other parts of the solver
            sg.volume_centroid[k] = V
            sg.wet_area_centroid[k] = A_w

            # Compute effective depth over wet area
            if A_w > 1.0e-12:
                h_eff = V / A_w
            else:
                h_eff = 0.0

            # Track difference between flat-bed and sub-grid height
            h_flat = max(0.0, eta - self.elev_c[k])
            total_correction += abs(h_eff - h_flat) * areas[k]

            # Update the height centroid value to reflect sub-grid terrain
            # Standard ANUGA: height = stage - elevation
            # Sub-grid: height = V / A_w (effective depth over wet area)
            self.height_c[k] = h_eff

            # If cell is very dry under sub-grid, kill momentum
            if h_eff < self.domain.minimum_allowed_height:
                self.xmom_c[k] = 0.0
                self.ymom_c[k] = 0.0

        self._total_volume_correction = total_correction

    def parallel_safe(self):
        """This operator is safe for parallel execution."""
        return True

    def statistics(self):
        msg = 'Sub-grid correction operator\n'
        msg += f'  Cells: {self.domain.number_of_elements}\n'
        if self.table is not None:
            valid = np.sum(self.table.n_breaks > 0)
            msg += f'  Cells with sub-grid tables: {valid}\n'
        return msg

    def timestepping_statistics(self):
        msg = f'  SubGrid: vol_correction={self._total_volume_correction:.6e}'
        return msg


class SubGridVolumeCorrector(Operator):
    """Volume-conserving sub-grid operator (more aggressive correction).

    This operator replaces the standard stage update with a volume-based
    update.  Rather than just updating the effective height, it:

    1. Computes V_old = V(stage_before_update)
    2. Computes dV = flux_divergence * dt * cell_area (from explicit_update)
    3. Sets V_new = V_old + dV
    4. Inverts: stage_new = V^{-1}(V_new)

    This requires running BEFORE the standard update_conserved_quantities,
    or storing the pre-update stage separately.

    For the MVP, this stores pre-update volumes and corrects after the
    standard update.

    Parameters
    ----------
    domain : anuga.Domain
        The domain with sub-grid data.
    """

    def __init__(self, domain, description=None, label=None,
                 logging=False, verbose=False):

        if description is None:
            description = 'Sub-grid volume-conserving update'
        if label is None:
            label = 'subgrid_volume'

        Operator.__init__(self, domain, description, label, logging, verbose)

        if not hasattr(domain, 'subgridData') or not domain.subgridData.enabled:
            raise RuntimeError(
                "SubGridVolumeCorrector requires domain.subgridData."
            )

        # Guard: warn if C-level sub-grid correction is also active,
        # since both would apply volume corrections causing double-correction.
        import warnings
        if getattr(domain, 'use_subgrid', False):
            warnings.warn(
                "SubGridVolumeCorrector: domain.use_subgrid is True, which "
                "means the C-level solver already applies sub-grid volume "
                "corrections. Using both will cause double-correction. "
                "Set domain.use_subgrid = False to use this Python operator "
                "instead, or don't create this operator.",
                RuntimeWarning,
                stacklevel=2
            )

        self.sg = domain.subgridData
        self.table = self.sg.cell_table

        n = domain.number_of_elements
        # Store pre-update volumes
        self._V_old = np.zeros(n, dtype=np.float64)
        self._stage_old = np.zeros(n, dtype=np.float64)
        self._initialized = False

    def store_pre_update_state(self):
        """Store volumes before the conserved quantity update.

        Call this from a hook before update_conserved_quantities().
        """
        stage_c = self.domain.quantities['stage'].centroid_values
        for k in range(self.domain.number_of_elements):
            self._stage_old[k] = stage_c[k]
            self._V_old[k] = self.table.volume_from_stage(stage_c[k], k)
        self._initialized = True

    def __call__(self):
        """Apply volume-conserving correction after standard update."""
        if not self._initialized:
            # First call: just compute current volumes for next step
            self.store_pre_update_state()
            return

        stage_c = self.stage_c
        areas = self.areas
        n = self.domain.number_of_elements
        table = self.table
        sg = self.sg

        for k in range(n):
            # The standard update changed stage by:
            #   dw = stage_new - stage_old
            # Interpret this as a volume change:
            #   dV = dw * cell_area
            dw = stage_c[k] - self._stage_old[k]
            dV = dw * areas[k]

            # Volume-conserving update
            V_new = self._V_old[k] + dV

            # Clamp to non-negative
            if V_new < 0.0:
                V_new = 0.0

            # Invert to get corrected stage
            stage_c[k] = table.stage_from_volume(V_new, k)

            # Update sub-grid state
            sg.volume_centroid[k] = V_new
            A_w = table.wet_area_from_stage(stage_c[k], k)
            sg.wet_area_centroid[k] = A_w

            # Effective depth
            if A_w > 1.0e-12:
                h_eff = V_new / A_w
            else:
                h_eff = 0.0

            self.height_c[k] = h_eff

            if h_eff < self.domain.minimum_allowed_height:
                self.xmom_c[k] = 0.0
                self.ymom_c[k] = 0.0

        # Store current state for next timestep
        self.store_pre_update_state()

    def parallel_safe(self):
        return True

    def statistics(self):
        msg = 'Sub-grid volume-conserving operator\n'
        msg += f'  Cells: {self.domain.number_of_elements}\n'
        return msg

    def timestepping_statistics(self):
        return '  SubGrid volume corrector active'
