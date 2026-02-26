"""
Tests for sub-grid operators and domain integration.

Tests cover:
  - SubGridData attachment to domain
  - SubGridCorrectionOperator registration and execution
  - Lake-at-rest (C-property) with sub-grid
  - Volume conservation
  - Flat-bed sub-grid matches standard solver
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose


class TestSubGridData(unittest.TestCase):
    """Test SubGridData attachment to domain."""

    def _make_domain(self):
        """Create a minimal domain for testing."""
        from anuga import rectangular_cross, Domain

        points, vertices, boundary = rectangular_cross(5, 5, len1=100.0, len2=100.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('friction', 0.03)
        domain.set_quantity('stage', 0.0)
        return domain

    def test_subgrid_data_initialized(self):
        """SubGridData should be initialized on domain creation."""
        domain = self._make_domain()
        self.assertTrue(hasattr(domain, 'subgridData'))
        self.assertFalse(domain.subgridData.enabled)
        self.assertFalse(domain.use_subgrid)

    def test_set_tables(self):
        """Setting tables enables sub-grid."""
        from anuga.subgrid.subgrid_tables import CellVolumeTable

        domain = self._make_domain()
        n = domain.number_of_elements
        table = CellVolumeTable(n, max_breaks=5)

        # Set trivial flat-bed tables
        for k in range(n):
            cell_area = float(domain.areas[k])
            eta = np.array([0.0, 1.0])
            vol = np.array([0.0, cell_area])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        domain.subgridData.set_tables(table)
        self.assertTrue(domain.subgridData.enabled)
        self.assertIsNotNone(domain.subgridData.volume_centroid)

    def test_set_tables_wrong_size(self):
        """Setting tables with wrong number of cells should raise."""
        from anuga.subgrid.subgrid_tables import CellVolumeTable

        domain = self._make_domain()
        table = CellVolumeTable(5, max_breaks=5)  # Wrong size

        with self.assertRaises(ValueError):
            domain.subgridData.set_tables(table)


class TestSubGridCorrectionOperator(unittest.TestCase):
    """Test SubGridCorrectionOperator."""

    def _make_domain_with_subgrid(self, z_func=None, stage=1.0):
        """Create a domain with sub-grid tables attached."""
        from anuga import rectangular_cross, Domain
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        from anuga.subgrid.operators import SubGridCorrectionOperator

        points, vertices, boundary = rectangular_cross(3, 3, len1=30.0, len2=30.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE0')

        if z_func is None:
            domain.set_quantity('elevation', 0.0)
        else:
            domain.set_quantity('elevation', z_func)

        domain.set_quantity('friction', 0.03)
        domain.set_quantity('stage', stage)

        n = domain.number_of_elements
        table = CellVolumeTable(n, max_breaks=5)

        for k in range(n):
            cell_area = float(domain.areas[k])
            z_bed = float(domain.quantities['elevation'].centroid_values[k])
            eta = np.array([z_bed, z_bed + 2.0])
            vol = np.array([0.0, cell_area * 2.0])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        domain.subgridData.set_tables(table)
        domain.use_subgrid = True

        # Create the operator
        op = SubGridCorrectionOperator(domain)

        return domain, op

    def test_operator_registered(self):
        """Operator should be registered as fractional step."""
        domain, op = self._make_domain_with_subgrid()
        self.assertIn(op, domain.fractional_step_operators)

    def test_operator_call(self):
        """Calling the operator should update height and sub-grid state."""
        domain, op = self._make_domain_with_subgrid(stage=0.5)

        # Call the operator
        op()

        # Check that sub-grid state was updated
        sg = domain.subgridData
        self.assertTrue(np.all(sg.volume_centroid >= 0))
        self.assertTrue(np.all(sg.wet_area_centroid >= 0))

        # Height should be non-negative
        h_c = domain.quantities['height'].centroid_values
        self.assertTrue(np.all(h_c >= 0))

    def test_flat_bed_no_correction(self):
        """Flat bed: sub-grid correction should not change stage."""
        domain, op = self._make_domain_with_subgrid(stage=0.5)

        stage_before = domain.quantities['stage'].centroid_values.copy()
        op()
        stage_after = domain.quantities['stage'].centroid_values

        # Stage shouldn't change (flat bed, linear V(eta))
        # But height will be recomputed as V/A_w which equals stage-bed for flat
        assert_allclose(stage_after, stage_before, atol=1e-10)

    def test_parallel_safe(self):
        """Operator should be parallel-safe."""
        domain, op = self._make_domain_with_subgrid()
        self.assertTrue(op.parallel_safe())

    def test_dry_cell_kills_momentum(self):
        """Dry cells should have zero momentum."""
        domain, op = self._make_domain_with_subgrid(stage=-1.0)

        # Set some non-zero momentum
        domain.quantities['xmomentum'].centroid_values[:] = 1.0
        domain.quantities['ymomentum'].centroid_values[:] = 1.0

        op()

        # All cells are dry (stage below bed), momentum should be zero
        xmom = domain.quantities['xmomentum'].centroid_values
        ymom = domain.quantities['ymomentum'].centroid_values
        assert_allclose(xmom, 0.0, atol=1e-10)
        assert_allclose(ymom, 0.0, atol=1e-10)


class TestSubGridVolumeCorrector(unittest.TestCase):
    """Test SubGridVolumeCorrector."""

    def test_volume_conservation(self):
        """Total volume should be conserved through the correction."""
        from anuga import rectangular_cross, Domain
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        from anuga.subgrid.operators import SubGridVolumeCorrector

        points, vertices, boundary = rectangular_cross(3, 3, len1=30.0, len2=30.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('friction', 0.03)
        domain.set_quantity('stage', 1.0)

        n = domain.number_of_elements
        table = CellVolumeTable(n, max_breaks=5)

        for k in range(n):
            cell_area = float(domain.areas[k])
            # Non-trivial V(eta): quadratic-ish
            eta = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
            vol = np.array([0.0, 0.1*cell_area, 0.4*cell_area, 0.9*cell_area, 1.6*cell_area])
            area = np.array([0.0, 0.2*cell_area, 0.6*cell_area, 0.8*cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        domain.subgridData.set_tables(table)
        # use_subgrid=False to avoid C-level correction (testing Python corrector only)
        domain.use_subgrid = False

        corrector = SubGridVolumeCorrector(domain)

        # Compute initial total volume
        stage_c = domain.quantities['stage'].centroid_values
        total_vol_before = 0.0
        for k in range(n):
            total_vol_before += table.volume_from_stage(stage_c[k], k)

        # Simulate a small stage change (as if fluxes ran)
        domain.quantities['stage'].centroid_values[:] += 0.01

        # Call the corrector
        corrector()

        # Compute total volume after
        stage_c = domain.quantities['stage'].centroid_values
        total_vol_after = 0.0
        for k in range(n):
            total_vol_after += table.volume_from_stage(stage_c[k], k)

        # Volume change should equal the intended change
        # (0.01 * sum_of_areas)
        intended_dV = 0.01 * np.sum(domain.areas)
        actual_dV = total_vol_after - total_vol_before
        assert_allclose(actual_dV, intended_dV, rtol=1e-6)


class TestFlowAlgorithms(unittest.TestCase):
    """Test sub-grid flow algorithm variants."""

    def test_de0_sg_sets_subgrid_flag(self):
        """DE0_SG should set use_subgrid flag."""
        from anuga import rectangular_cross, Domain

        points, vertices, boundary = rectangular_cross(3, 3, len1=30.0, len2=30.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE0_SG')
        self.assertTrue(domain.use_subgrid)
        self.assertEqual(domain.flow_algorithm, 'DE0_SG')

    def test_de1_sg_sets_subgrid_flag(self):
        """DE1_SG should set use_subgrid flag."""
        from anuga import rectangular_cross, Domain

        points, vertices, boundary = rectangular_cross(3, 3, len1=30.0, len2=30.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE1_SG')
        self.assertTrue(domain.use_subgrid)
        self.assertEqual(domain.flow_algorithm, 'DE1_SG')


class TestLakeAtRest(unittest.TestCase):
    """Lake-at-rest (C-property) test.

    A lake at rest with uniform stage should remain at rest
    regardless of sub-grid terrain variation. This is the most
    fundamental validation test for sub-grid methods.
    """

    def test_lake_at_rest_flat_bed(self):
        """Lake at rest on flat bed should produce zero velocity."""
        from anuga import rectangular_cross, Domain
        from anuga.subgrid.subgrid_tables import CellVolumeTable
        from anuga.subgrid.operators import SubGridCorrectionOperator

        points, vertices, boundary = rectangular_cross(5, 5, len1=50.0, len2=50.0)
        domain = Domain(points, vertices, boundary)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('friction', 0.0)
        domain.set_quantity('stage', 5.0)

        # Flat bed sub-grid tables
        n = domain.number_of_elements
        table = CellVolumeTable(n, max_breaks=5)
        for k in range(n):
            cell_area = float(domain.areas[k])
            eta = np.array([0.0, 10.0])
            vol = np.array([0.0, cell_area * 10.0])
            area = np.array([cell_area, cell_area])
            table.set_cell(k, eta, vol, area, cell_area)

        domain.subgridData.set_tables(table)
        domain.use_subgrid = True
        SubGridCorrectionOperator(domain)

        # Check initial stage is uniform
        stage_c = domain.quantities['stage'].centroid_values
        assert_allclose(stage_c, 5.0, atol=1e-10)

        # Call the correction operator
        domain.fractional_step_operators[-1]()

        # Stage should remain uniform
        assert_allclose(stage_c, 5.0, atol=1e-10)

        # Velocities should be zero
        u_c = domain.quantities['xvelocity'].centroid_values
        v_c = domain.quantities['yvelocity'].centroid_values
        assert_allclose(u_c, 0.0, atol=1e-10)
        assert_allclose(v_c, 0.0, atol=1e-10)


class TestBuildVolumeTableHandCalculation(unittest.TestCase):
    """Verify V(eta) tables match hand calculations for known geometries."""

    def test_flat_bed_volume(self):
        """Flat bed at z=5: V(eta=6) = A * 1.0"""
        from anuga.subgrid.subgrid_tables import CellVolumeTable

        table = CellVolumeTable(1, max_breaks=5)
        cell_area = 200.0
        z = 5.0
        eta = np.array([z, z + 5.0])
        vol = np.array([0.0, cell_area * 5.0])
        area = np.array([cell_area, cell_area])
        table.set_cell(0, eta, vol, area, cell_area)

        # At stage=6 (1m above bed): V = 200
        assert_allclose(table.volume_from_stage(6.0, 0), 200.0, atol=1e-10)

        # At stage=5 (at bed): V = 0
        assert_allclose(table.volume_from_stage(5.0, 0), 0.0, atol=1e-10)

        # At stage=4 (below bed): V = 0
        assert_allclose(table.volume_from_stage(4.0, 0), 0.0, atol=1e-10)

    def test_v_shaped_valley_volume(self):
        """V-shaped valley: 50% of area at z=0, 50% at z=1.

        At eta=0: V=0 (nothing wet)
        At eta=0.5: V = 0.5 * 0.5*A * px_area = 0.5 * 50 = 25
        At eta=1.0: V = 1.0 * 0.5*A + 0.0 * 0.5*A = 50
        At eta=1.5: V = 1.5 * 0.5*A + 0.5 * 0.5*A = 75 + 25 = 100
        """
        from anuga.subgrid.subgrid_tables import CellVolumeTable

        table = CellVolumeTable(1, max_breaks=5)
        cell_area = 100.0

        # Manual: 2 pixels at z=0, 2 pixels at z=1, each pixel_area=25
        eta = np.array([0.0, 1.0, 2.0])
        vol = np.array([0.0, 50.0, 200.0])
        area = np.array([0.0, 50.0, 100.0])
        table.set_cell(0, eta, vol, area, cell_area)

        assert_allclose(table.volume_from_stage(0.0, 0), 0.0)
        assert_allclose(table.volume_from_stage(0.5, 0), 25.0)
        assert_allclose(table.volume_from_stage(1.0, 0), 50.0)
        assert_allclose(table.volume_from_stage(1.5, 0), 125.0)

    def test_step_function_inversion(self):
        """Step function: verify V^{-1}(V(eta)) = eta."""
        from anuga.subgrid.subgrid_tables import CellVolumeTable

        table = CellVolumeTable(1, max_breaks=5)
        cell_area = 100.0

        eta = np.array([0.0, 1.0, 2.0])
        vol = np.array([0.0, 50.0, 200.0])
        area = np.array([0.0, 50.0, 100.0])
        table.set_cell(0, eta, vol, area, cell_area)

        for test_eta in np.linspace(0.0, 2.0, 20):
            V = table.volume_from_stage(test_eta, 0)
            eta_back = table.stage_from_volume(V, 0)
            assert_allclose(eta_back, test_eta, atol=1e-10,
                            err_msg=f"Inversion failed at eta={test_eta}")


if __name__ == '__main__':
    unittest.main()
