"""
Regression tests for the SGS "building cell" phantom-depth fix.

Background
----------
When mesh elevation smoothing (alpha close to 1) is used, cells near buildings
can end up with ``z_min > bed_centroid``.  In those cells, two things go wrong
without the fix:

1. The C ``_openmp_protect`` function enforces ``stage >= z_min``.  Since
   ``z_min > bed_centroid``, the dry state has ``stage = z_min > bed_centroid``,
   which looks like a wet cell with flat-bed depth = ``z_min - bed_centroid``.
   This creates phantom mass that grows without bound.

2. ``sg_stage_from_volume(tiny_V)`` returns ``~z_min`` (38 m in the worst case)
   even for a tiny volume, because all DEM data inside the triangle lies above
   ``z_min``.  Any real water entering the cell triggers an enormous phantom
   stage jump.

Fix applied in ``set_subgrid_dem()`` (shallow_water_domain.py):

  Fix 1: For cells where z_min > bed_centroid, replace the SGS table with a
         two-breakpoint flat-bed table anchored at bed_centroid so that
         z_min = bed_centroid and V(stage) = (stage - bed_centroid) * area.

  Fix 2: For any cell where stage > z_min and stage <= bed_centroid (a cell
         that looks sub-grid-wet but flat-bed-dry), reset stage = z_min so
         that V_old = 0 before the first timestep.

These tests verify both fixes by directly manipulating a CellVolumeTable
object and applying the same logic, without requiring a real DEM file.
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose


def _apply_building_cell_fix(cell_table, bed_cv, stage_cv, FLATBED_RANGE=60.0):
    """Apply the phantom-depth fix to a CellVolumeTable.

    This mirrors the fix logic in ``set_subgrid_dem()`` and is used in tests
    so we can verify the outcome without needing a full Domain + DEM file.

    Parameters
    ----------
    cell_table : CellVolumeTable
    bed_cv     : array of bed centroid elevations (m), length n_cells
    stage_cv   : array of stage centroid values (m), length n_cells (modified in-place)
    FLATBED_RANGE : float, height range for the replacement flat-bed table (m)
    """
    z_min = cell_table.z_min

    # Fix 1: replace table for building cells (z_min > bed_centroid)
    problem_cells = z_min > bed_cv
    if problem_cells.any():
        for k in np.where(problem_cells)[0]:
            bed_k  = float(bed_cv[k])
            area_k = float(cell_table.cell_areas[k])
            cell_table.eta_breaks[k, :] = np.nan
            cell_table.vol_cumul[k,  :] = np.nan
            cell_table.wet_area[k,   :] = np.nan
            cell_table.eta_breaks[k, 0] = bed_k
            cell_table.eta_breaks[k, 1] = bed_k + FLATBED_RANGE
            cell_table.vol_cumul[k,  0] = 0.0
            cell_table.vol_cumul[k,  1] = FLATBED_RANGE * area_k
            cell_table.wet_area[k,   0] = area_k
            cell_table.wet_area[k,   1] = area_k
            cell_table.n_breaks[k] = 2
        cell_table.z_min[problem_cells] = bed_cv[problem_cells]

    # Fix 2: correct phantom stage for sub-grid-wet / flat-bed-dry cells
    phantom = (stage_cv > cell_table.z_min) & (stage_cv <= bed_cv)
    stage_cv[phantom] = cell_table.z_min[phantom]


class TestBuildingCellFix(unittest.TestCase):
    """Tests for the z_min > bed_centroid phantom-depth fix."""

    def _make_table(self, n_cells=4, max_breaks=10):
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        return CellVolumeTable(n_cells, max_breaks=max_breaks)

    def _setup_normal_cell(self, table, k, z_bed=5.0, cell_area=100.0, n_breaks=3):
        """Set up a normal cell (z_min == z_bed) with a simple V(eta) curve."""
        eta  = np.linspace(z_bed, z_bed + 2.0, n_breaks)
        vol  = np.array([(e - z_bed) * cell_area for e in eta])
        area = np.full(n_breaks, cell_area)
        table.set_cell(k, eta, vol, area, cell_area)

    def _setup_building_cell(self, table, k, z_min=40.0, cell_area=100.0, n_breaks=3):
        """Set up a 'building cell' where the DEM is entirely above terrain floor.

        The DEM data inside the cell ranges from z_min (building floor) up to
        z_min + 2.  The coarse mesh bed_centroid is lower (set separately).
        """
        eta  = np.linspace(z_min, z_min + 2.0, n_breaks)
        vol  = np.array([(e - z_min) * cell_area for e in eta])
        area = np.full(n_breaks, cell_area)
        table.set_cell(k, eta, vol, area, cell_area)

    # ── Fix 1 tests ────────────────────────────────────────────────────────

    def test_fix1_z_min_corrected(self):
        """After fix 1: z_min for building cells equals bed_centroid."""
        table = self._make_table(n_cells=2)
        self._setup_normal_cell(table,   k=0, z_bed=5.0,  cell_area=100.0)
        self._setup_building_cell(table, k=1, z_min=40.0, cell_area=100.0)

        bed_cv   = np.array([5.0, 20.0])   # k=1 has z_min=40 > bed=20
        stage_cv = np.array([5.0, 20.0])

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # k=0: normal cell unchanged
        self.assertAlmostEqual(table.z_min[0], 5.0)
        # k=1: z_min lowered to bed_centroid
        self.assertAlmostEqual(table.z_min[1], 20.0)

    def test_fix1_no_phantom_volume_at_dry_state(self):
        """After fix 1: volume at z_min equals zero (no phantom water)."""
        table = self._make_table(n_cells=1)
        self._setup_building_cell(table, k=0, z_min=40.0, cell_area=100.0)

        bed_cv   = np.array([20.0])  # z_min=40 > bed=20 → building cell
        stage_cv = np.array([20.0])

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # After fix, z_min == bed_cv == 20.0
        # Volume at z_min must be 0
        v_at_zmin = table.volume_from_stage(table.z_min[0], 0)
        self.assertAlmostEqual(v_at_zmin, 0.0)

    def test_fix1_flat_bed_behaviour_after_replacement(self):
        """After fix 1: building cell has correct flat-bed V(eta) = area*(eta-bed)."""
        cell_area = 100.0
        z_bed = 20.0
        FLATBED_RANGE = 60.0

        table = self._make_table(n_cells=1)
        self._setup_building_cell(table, k=0, z_min=40.0, cell_area=cell_area)

        bed_cv   = np.array([z_bed])
        stage_cv = np.array([z_bed])
        _apply_building_cell_fix(table, bed_cv, stage_cv, FLATBED_RANGE=FLATBED_RANGE)

        # Test V(eta) = area*(eta - z_bed) for eta >= z_bed
        for depth in [0.0, 1.0, 5.0, 30.0]:
            stage = z_bed + depth
            expected_vol = cell_area * depth
            actual_vol = table.volume_from_stage(stage, 0)
            assert_allclose(actual_vol, expected_vol, rtol=1e-6,
                            err_msg=f"V at depth={depth}m should be {expected_vol}")

    def test_fix1_normal_cells_unaffected(self):
        """Fix 1 must not change tables for cells where z_min <= bed_centroid."""
        table = self._make_table(n_cells=3)
        cell_area = 100.0
        for k, z_bed in enumerate([0.0, 5.0, 10.0]):
            self._setup_normal_cell(table, k, z_bed=z_bed, cell_area=cell_area)

        bed_cv   = np.array([0.0, 5.0, 10.0])
        stage_cv = np.array([0.0, 5.0, 10.0])

        # Record volumes before fix
        vols_before = [table.volume_from_stage(bed_cv[k] + 0.5, k) for k in range(3)]

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # Normal cells must be unchanged
        vols_after = [table.volume_from_stage(bed_cv[k] + 0.5, k) for k in range(3)]
        for k in range(3):
            assert_allclose(vols_after[k], vols_before[k], rtol=1e-10,
                            err_msg=f"Cell {k} (normal) should not be modified")

    def test_fix1_n_breaks_set_to_2(self):
        """After fix 1: building cell has exactly 2 breakpoints."""
        table = self._make_table(n_cells=1, max_breaks=20)
        self._setup_building_cell(table, k=0, z_min=40.0, n_breaks=10,
                                  cell_area=100.0)

        bed_cv   = np.array([20.0])
        stage_cv = np.array([20.0])
        _apply_building_cell_fix(table, bed_cv, stage_cv)

        self.assertEqual(int(table.n_breaks[0]), 2)

    # ── Fix 2 tests ────────────────────────────────────────────────────────

    def test_fix2_phantom_stage_corrected(self):
        """Fix 2: stage is reset to z_min when stage > z_min and stage <= bed."""
        table = self._make_table(n_cells=2)
        cell_area = 100.0
        # Cell 0: normal cell, z_min == bed == 5.0, stage == 6.0 (genuinely wet)
        self._setup_normal_cell(table, k=0, z_bed=5.0, cell_area=cell_area)
        # Cell 1: sub-grid channel cell: z_min=3.0 < bed=5.0, stage=5.0 (flat-bed dry)
        #         This is the phantom: stage=5.0 > z_min=3.0 but stage <= bed=5.0
        eta  = np.array([3.0, 5.0, 7.0])
        vol  = np.array([0.0, cell_area * 2.0, cell_area * 4.0])
        area = np.full(3, cell_area)
        table.set_cell(1, eta, vol, area, cell_area)

        bed_cv   = np.array([5.0, 5.0])
        stage_cv = np.array([6.0, 5.0])  # k=1: stage=bed (phantom wet in SGS sense)

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # k=0: genuine wet cell, stage unchanged
        self.assertAlmostEqual(stage_cv[0], 6.0)
        # k=1: phantom state corrected to z_min
        self.assertAlmostEqual(stage_cv[1], 3.0)  # = z_min[1] = 3.0

    def test_fix2_already_dry_not_touched(self):
        """Fix 2: cells where stage == z_min (correctly dry) are not modified."""
        table = self._make_table(n_cells=1)
        self._setup_normal_cell(table, k=0, z_bed=5.0, cell_area=100.0)

        bed_cv   = np.array([5.0])
        stage_cv = np.array([5.0])  # stage == z_min == bed: correctly dry

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # Stage should remain 5.0
        self.assertAlmostEqual(stage_cv[0], 5.0)

    # ── Combined fix tests ──────────────────────────────────────────────────

    def test_combined_merewether_scenario(self):
        """Simulates a worst-case Merewether building cell.

        A cell with z_min=38 m (building top in DEM), bed_centroid=18 m
        (heavily smoothed mesh) must have zero phantom volume after the fix,
        and stage must be at bed_centroid (18 m) = z_min (after fix).
        """
        cell_area = 16.0  # typical 4m mesh cell area (m²)
        z_min_building = 38.0
        z_bed = 18.0

        table = self._make_table(n_cells=1, max_breaks=20)
        # Build a realistic building DEM table: terrain starts at 38 m
        eta  = np.linspace(z_min_building, z_min_building + 40.0, 20)
        vol  = np.array([(e - z_min_building) * cell_area for e in eta])
        area = np.full(20, cell_area)
        table.set_cell(0, eta, vol, area, cell_area)

        # Before fix: z_min=38 > bed=18 → building cell condition present.
        # The phantom isn't in volume_from_stage(bed_cv) — that correctly returns 0.
        # The phantom is that _openmp_protect would enforce stage = z_min = 38,
        # giving flat-bed depth = 38 - 18 = 20 m.  Verify the precondition:
        self.assertGreater(table.z_min[0], z_bed,
                           "Before fix: z_min must exceed bed_centroid")
        # Confirm that V(z_min) = 0 at the DEM floor (correct for table)
        self.assertAlmostEqual(table.volume_from_stage(z_min_building, 0), 0.0)
        # Confirm the phantom implied depth = z_min - bed > 0 before fix
        phantom_depth_before = table.z_min[0] - z_bed
        self.assertGreater(phantom_depth_before, 0.0,
                           "Before fix: implied phantom depth must be positive")

        bed_cv   = np.array([z_bed])
        stage_cv = np.array([z_bed])  # initially set to bed_centroid (ANUGA default)
        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # After fix: z_min == bed_centroid (phantom depth = 0)
        self.assertAlmostEqual(table.z_min[0], z_bed)
        phantom_depth_after = table.z_min[0] - z_bed
        self.assertAlmostEqual(phantom_depth_after, 0.0,
                               msg="After fix: no phantom depth (z_min == bed)")
        # After fix: V(z_min) = 0
        v_at_new_zmin = table.volume_from_stage(table.z_min[0], 0)
        self.assertAlmostEqual(v_at_new_zmin, 0.0,
                               msg="After fix: zero phantom volume at dry state")

    def test_multiple_building_cells(self):
        """333 building cells (as found in Merewether 2m mesh) all get fixed."""
        n_normal   = 200
        n_building = 333
        n_cells = n_normal + n_building
        cell_area = 4.0  # 2m mesh cell

        table = self._make_table(n_cells=n_cells, max_breaks=5)
        bed_cv   = np.zeros(n_cells)
        stage_cv = np.zeros(n_cells)

        for k in range(n_normal):
            self._setup_normal_cell(table, k, z_bed=10.0, cell_area=cell_area)
            bed_cv[k]   = 10.0
            stage_cv[k] = 10.0

        for k in range(n_normal, n_cells):
            self._setup_building_cell(table, k, z_min=40.0, cell_area=cell_area)
            bed_cv[k]   = 20.0   # z_min=40 > bed=20 → building cell
            stage_cv[k] = 20.0

        _apply_building_cell_fix(table, bed_cv, stage_cv)

        # All building cells must have z_min == bed_centroid
        for k in range(n_normal, n_cells):
            self.assertAlmostEqual(
                table.z_min[k], bed_cv[k],
                msg=f"Building cell {k}: z_min should be bed_centroid after fix"
            )

        # No building cell should have non-zero volume at its z_min
        for k in range(n_normal, n_cells):
            v = table.volume_from_stage(table.z_min[k], k)
            self.assertAlmostEqual(
                v, 0.0,
                msg=f"Building cell {k}: phantom volume must be zero after fix"
            )

        # Normal cells must be unchanged (z_min still 10.0)
        for k in range(n_normal):
            self.assertAlmostEqual(table.z_min[k], 10.0)


if __name__ == '__main__':
    unittest.main()
