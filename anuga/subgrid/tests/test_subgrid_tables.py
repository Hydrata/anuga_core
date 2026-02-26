"""
Tests for sub-grid lookup tables.

Tests cover:
  - CellVolumeTable construction and interpolation
  - Volume-stage inversion (round-trip)
  - EdgeAreaTable construction
  - Flat bed (trivial) case
  - V-shaped valley
  - Step function
  - Piecewise-linear consistency
  - Volume conservation
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose


class TestCellVolumeTableBasic(unittest.TestCase):
    """Basic CellVolumeTable operations."""

    def setUp(self):
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        self.CellVolumeTable = CellVolumeTable

    def test_flat_bed(self):
        """Flat bed: V(eta) = cell_area * (eta - z_bed) for eta > z_bed."""
        table = self.CellVolumeTable(1, max_breaks=10)
        z_bed = 5.0
        cell_area = 100.0

        eta = np.array([z_bed, z_bed + 1.0])
        vol = np.array([0.0, cell_area])
        area = np.array([cell_area, cell_area])
        table.set_cell(0, eta, vol, area, cell_area)

        # Test at breakpoints
        assert_allclose(table.volume_from_stage(z_bed, 0), 0.0)
        assert_allclose(table.volume_from_stage(z_bed + 1.0, 0), cell_area)

        # Test interpolation
        assert_allclose(table.volume_from_stage(z_bed + 0.5, 0), 50.0)

        # Test extrapolation above
        assert_allclose(table.volume_from_stage(z_bed + 2.0, 0), 200.0)

        # Test below bed
        assert_allclose(table.volume_from_stage(z_bed - 1.0, 0), 0.0)

    def test_v_shaped_valley(self):
        """V-shaped valley: volume increases quadratically."""
        table = self.CellVolumeTable(1, max_breaks=5)
        cell_area = 100.0

        # Simulated: 4 sample points at z = [0, 1, 1, 2]
        # At eta=0: V=0, A=0
        # At eta=1: V = (1-0)*px = 25, A=25
        # At eta=2: V = (2-0)*25 + (2-1)*50 = 50+50=100, A=100
        eta = np.array([0.0, 1.0, 2.0])
        vol = np.array([0.0, 25.0, 100.0])
        area = np.array([0.0, 25.0, 100.0])
        table.set_cell(0, eta, vol, area, cell_area)

        # Mid-point interpolation
        V_mid = table.volume_from_stage(0.5, 0)
        assert_allclose(V_mid, 12.5)

    def test_inversion_roundtrip(self):
        """V(eta) -> V -> eta round-trip should be exact."""
        table = self.CellVolumeTable(1, max_breaks=10)
        cell_area = 200.0

        # Irregular V(eta) curve
        eta = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        vol = np.array([0.0, 10.0, 40.0, 90.0, 160.0, 350.0])
        area = np.array([0.0, 40.0, 80.0, 120.0, 160.0, 200.0])
        table.set_cell(0, eta, vol, area, cell_area)

        # Test at breakpoints
        for i in range(len(eta)):
            V = table.volume_from_stage(eta[i], 0)
            eta_inv = table.stage_from_volume(V, 0)
            assert_allclose(eta_inv, eta[i], atol=1e-12,
                            err_msg=f"Round-trip failed at breakpoint {i}")

        # Test at intermediate points
        for test_eta in [0.25, 0.75, 1.25, 2.5]:
            V = table.volume_from_stage(test_eta, 0)
            eta_inv = table.stage_from_volume(V, 0)
            assert_allclose(eta_inv, test_eta, atol=1e-12,
                            err_msg=f"Round-trip failed at eta={test_eta}")

    def test_wet_area_interpolation(self):
        """Wet area interpolation."""
        table = self.CellVolumeTable(1, max_breaks=5)
        cell_area = 100.0
        eta = np.array([0.0, 1.0, 2.0])
        vol = np.array([0.0, 25.0, 100.0])
        area = np.array([0.0, 50.0, 100.0])
        table.set_cell(0, eta, vol, area, cell_area)

        assert_allclose(table.wet_area_from_stage(-1.0, 0), 0.0)
        assert_allclose(table.wet_area_from_stage(0.0, 0), 0.0)
        assert_allclose(table.wet_area_from_stage(0.5, 0), 25.0)
        assert_allclose(table.wet_area_from_stage(1.0, 0), 50.0)
        assert_allclose(table.wet_area_from_stage(1.5, 0), 75.0)
        assert_allclose(table.wet_area_from_stage(2.0, 0), 100.0)
        # Above max: returns full cell area
        assert_allclose(table.wet_area_from_stage(3.0, 0), cell_area)

    def test_empty_cell(self):
        """Cell with no breakpoints returns zero."""
        table = self.CellVolumeTable(1, max_breaks=5)
        assert_allclose(table.volume_from_stage(10.0, 0), 0.0)
        assert_allclose(table.wet_area_from_stage(10.0, 0), 0.0)

    def test_multi_cell(self):
        """Multiple cells with different tables."""
        table = self.CellVolumeTable(3, max_breaks=5)

        for k in range(3):
            z = float(k)
            cell_area = 100.0 * (k + 1)
            eta = np.array([z, z + 1.0])
            vol = np.array([0.0, cell_area])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        # Cell 0: z=0, area=100
        assert_allclose(table.volume_from_stage(0.5, 0), 50.0)
        # Cell 1: z=1, area=200
        assert_allclose(table.volume_from_stage(1.5, 1), 100.0)
        # Cell 2: z=2, area=300
        assert_allclose(table.volume_from_stage(2.5, 2), 150.0)

    def test_vectorized_volume(self):
        """Vectorized volume_from_stage_vec matches per-cell results."""
        table = self.CellVolumeTable(3, max_breaks=5)
        cell_area = 100.0
        for k in range(3):
            z = float(k)
            eta = np.array([z, z + 2.0])
            vol = np.array([0.0, 2.0 * cell_area])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        stages = np.array([0.5, 1.5, 3.0])
        V_vec = table.volume_from_stage_vec(stages)

        for k in range(3):
            V_scalar = table.volume_from_stage(stages[k], k)
            assert_allclose(V_vec[k], V_scalar)

    def test_vectorized_inversion(self):
        """Vectorized stage_from_volume_vec round-trips correctly."""
        table = self.CellVolumeTable(2, max_breaks=5)
        cell_area = 100.0
        for k in range(2):
            z = float(k)
            eta = np.array([z, z + 1.0])
            vol = np.array([0.0, cell_area])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        stages = np.array([0.5, 1.5])
        V = table.volume_from_stage_vec(stages)
        stages_back = table.stage_from_volume_vec(V)
        assert_allclose(stages_back, stages, atol=1e-12)


class TestCellVolumeTablePhysics(unittest.TestCase):
    """Physics-based tests for CellVolumeTable."""

    def setUp(self):
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        self.CellVolumeTable = CellVolumeTable

    def test_monotonicity(self):
        """V(eta) must be monotonically non-decreasing."""
        table = self.CellVolumeTable(1, max_breaks=20)
        cell_area = 100.0

        # Random elevation profile
        np.random.seed(42)
        n_bp = 15
        eta = np.sort(np.random.uniform(0, 10, n_bp))
        vol = np.cumsum(np.random.uniform(0, 10, n_bp))
        vol[0] = 0.0
        area = np.linspace(10, cell_area, n_bp)
        table.set_cell(0, eta, vol, area, cell_area)

        # Sample at 100 points
        test_etas = np.linspace(eta[0] - 1, eta[-1] + 1, 100)
        V_prev = -np.inf
        for te in test_etas:
            V = table.volume_from_stage(te, 0)
            self.assertGreaterEqual(V, V_prev,
                                    f"V(eta) not monotonic at eta={te}")
            V_prev = V

    def test_c_property_flat_bed(self):
        """Lake at rest on a flat bed: V(eta) = A * (eta - z)."""
        table = self.CellVolumeTable(1, max_breaks=5)
        z = 3.0
        A = 150.0
        eta = np.array([z, z + 5.0])
        vol = np.array([0.0, A * 5.0])
        area = np.array([A, A])
        table.set_cell(0, eta, vol, area, A)

        # At any stage above bed: V should be A * (eta - z)
        for test_eta in [3.0, 3.5, 5.0, 8.0, 10.0]:
            expected = A * max(0, test_eta - z)
            assert_allclose(table.volume_from_stage(test_eta, 0), expected,
                            atol=1e-10)


class TestEdgeAreaTable(unittest.TestCase):
    """EdgeAreaTable basic tests."""

    def setUp(self):
        from anuga.subgrid.subgrid_tables import EdgeAreaTable
        self.EdgeAreaTable = EdgeAreaTable

    def test_flat_edge(self):
        """Flat edge: flow area = edge_length * (eta - z)."""
        table = self.EdgeAreaTable(1, max_breaks=5)
        z = 2.0
        L = 10.0
        eta = np.array([z, z + 1.0])
        area = np.array([0.0, L])
        width = np.array([L, L])
        table.set_edge(0, eta, area, width)

        assert_allclose(table.flow_area_from_stage(z, 0), 0.0)
        assert_allclose(table.flow_area_from_stage(z + 0.5, 0), 5.0)
        assert_allclose(table.flow_area_from_stage(z + 1.0, 0), 10.0)

        # Extrapolation
        assert_allclose(table.flow_area_from_stage(z + 2.0, 0), 20.0)

    def test_flow_width(self):
        """Flow width interpolation."""
        table = self.EdgeAreaTable(1, max_breaks=5)
        eta = np.array([0.0, 1.0, 2.0])
        area = np.array([0.0, 2.0, 8.0])
        width = np.array([0.0, 4.0, 8.0])
        table.set_edge(0, eta, area, width)

        assert_allclose(table.flow_width_from_stage(-1.0, 0), 0.0)
        assert_allclose(table.flow_width_from_stage(0.5, 0), 2.0)
        assert_allclose(table.flow_width_from_stage(1.0, 0), 4.0)
        assert_allclose(table.flow_width_from_stage(1.5, 0), 6.0)
        # Above max: returns last width
        assert_allclose(table.flow_width_from_stage(3.0, 0), 8.0)


class TestBuildVolumeTable(unittest.TestCase):
    """Tests for _build_volume_table."""

    def setUp(self):
        from anuga.subgrid.dem_sampler import _build_volume_table
        self._build_volume_table = _build_volume_table

    def test_flat_bed_samples(self):
        """All sample points at same elevation -> flat bed table."""
        elevations = np.array([5.0, 5.0, 5.0, 5.0])
        pixel_area = 25.0
        cell_area = 100.0

        eta, vol, area = self._build_volume_table(elevations, pixel_area, 10, cell_area)

        # Flat bed should have 2 breakpoints
        self.assertEqual(len(eta), 2)
        assert_allclose(vol[0], 0.0)
        assert_allclose(area[0], cell_area)

    def test_step_function(self):
        """Half the cell at z=0, half at z=1."""
        elevations = np.array([0.0, 0.0, 1.0, 1.0])
        pixel_area = 25.0
        cell_area = 100.0

        eta, vol, area = self._build_volume_table(elevations, pixel_area, 10, cell_area)

        # At eta=0 (z_min): V=0, A=0
        assert_allclose(vol[0], 0.0)
        assert_allclose(area[0], 0.0)

        # At eta=1: first 2 pixels fully wet, V = 2 * (1-0) * 25 = 50
        V_at_1 = np.interp(1.0, eta, vol)
        self.assertAlmostEqual(V_at_1, 50.0, places=1)

    def test_volume_non_negative(self):
        """Volume should never be negative."""
        np.random.seed(123)
        elevations = np.random.uniform(0, 10, 50)
        pixel_area = 2.0
        cell_area = 100.0

        eta, vol, area = self._build_volume_table(elevations, pixel_area, 20, cell_area)
        self.assertTrue(np.all(vol >= 0), "Negative volume in table")

    def test_area_bounded(self):
        """Wet area should be between 0 and cell_area."""
        np.random.seed(456)
        elevations = np.random.uniform(0, 5, 30)
        pixel_area = 3.33
        cell_area = 100.0

        eta, vol, area = self._build_volume_table(elevations, pixel_area, 15, cell_area)
        self.assertTrue(np.all(area >= 0), "Negative area in table")
        self.assertTrue(np.all(area <= cell_area + 1e-10), "Area exceeds cell area")


class TestSubgridIO(unittest.TestCase):
    """Tests for save/load round-trip."""

    def test_save_load_roundtrip(self):
        """Save and load cell tables, verify data integrity."""
        import tempfile
        import os
        from anuga.subgrid.subgrid_tables import CellVolumeTable, EdgeAreaTable
        from anuga.subgrid.subgrid_io import save_subgrid_tables, load_subgrid_tables

        # Create tables
        cell_table = CellVolumeTable(2, max_breaks=5)
        for k in range(2):
            z = float(k)
            eta = np.array([z, z + 1.0, z + 2.0])
            vol = np.array([0.0, 50.0, 200.0])
            area = np.array([0.0, 50.0, 100.0])
            cell_table.set_cell(k, eta, vol, area, 100.0)

        edge_table = EdgeAreaTable(6, max_breaks=5)
        for e in range(6):
            eta = np.array([0.0, 1.0])
            area = np.array([0.0, 10.0])
            width = np.array([10.0, 10.0])
            edge_table.set_edge(e, eta, area, width)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_tables')
            save_subgrid_tables(filepath, cell_table, edge_table)
            cell_loaded, edge_loaded = load_subgrid_tables(filepath)

        # Verify cell data
        self.assertEqual(cell_loaded.n_cells, 2)
        assert_allclose(cell_loaded.eta_breaks, cell_table.eta_breaks)
        assert_allclose(cell_loaded.vol_cumul, cell_table.vol_cumul)
        assert_allclose(cell_loaded.wet_area, cell_table.wet_area)
        assert_allclose(cell_loaded.n_breaks, cell_table.n_breaks)
        assert_allclose(cell_loaded.z_min, cell_table.z_min)
        assert_allclose(cell_loaded.z_max, cell_table.z_max)

        # Verify edge data
        self.assertIsNotNone(edge_loaded)
        self.assertEqual(edge_loaded.n_edges, 6)
        assert_allclose(edge_loaded.eta_breaks, edge_table.eta_breaks)
        assert_allclose(edge_loaded.flow_area, edge_table.flow_area)

    def test_save_load_without_edges(self):
        """Save and load cell-only tables."""
        import tempfile
        import os
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        from anuga.subgrid.subgrid_io import save_subgrid_tables, load_subgrid_tables

        cell_table = CellVolumeTable(1, max_breaks=3)
        eta = np.array([0.0, 1.0, 2.0])
        vol = np.array([0.0, 50.0, 150.0])
        area = np.array([0.0, 50.0, 100.0])
        cell_table.set_cell(0, eta, vol, area, 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'cell_only')
            save_subgrid_tables(filepath, cell_table)
            cell_loaded, edge_loaded = load_subgrid_tables(filepath)

        self.assertIsNone(edge_loaded)
        assert_allclose(cell_loaded.vol_cumul, cell_table.vol_cumul)


if __name__ == '__main__':
    unittest.main()
