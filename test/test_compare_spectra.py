import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperlyse.database import Database


def assert_equal(old, new, label):
    """Assert that old and new results are numerically identical."""
    if old is None and new is None:
        return
    if old is None or new is None:
        pytest.fail(f"{label}: one is None, the other is not (old={old}, new={new})")
    old = np.asarray(old)
    new = np.asarray(new)
    assert old.shape == new.shape, f"{label}: shape mismatch old={old.shape} new={new.shape}"
    np.testing.assert_allclose(new, old, rtol=0, atol=0,
                               err_msg=f"{label}: values differ")


# ---------------------------------------------------------------------------
# 1D spectrum tests — same wavelength grid
# ---------------------------------------------------------------------------

class TestSameGrid1D:
    """Both spectra share the exact same wavelength array."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(0)
        self.x1 = np.linspace(400, 800, 100)
        self.y1 = np.random.rand(100)
        self.x2 = np.linspace(400, 800, 100)
        self.y2 = np.random.rand(100)

    def test_squared_errors(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           squared_errs=True)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       squared_errs=True)
        assert_equal(old, new, "1D same grid squared")

    def test_abs_errors(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       squared_errs=False)
        assert_equal(old, new, "1D same grid abs")

    def test_gradient_squared(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           use_gradient=True, squared_errs=True)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       use_gradient=True, squared_errs=True)
        assert_equal(old, new, "1D same grid gradient squared")

    def test_gradient_abs(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           use_gradient=True, squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       use_gradient=True, squared_errs=False)
        assert_equal(old, new, "1D same grid gradient abs")

    def test_custom_range(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           custom_range=cr)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       custom_range=cr)
        assert_equal(old, new, "1D same grid custom_range")

    def test_custom_range_gradient_abs(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           custom_range=cr, use_gradient=True,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       custom_range=cr, use_gradient=True,
                                       squared_errs=False)
        assert_equal(old, new, "1D same grid custom_range+gradient+abs")


# ---------------------------------------------------------------------------
# 1D spectrum tests — different wavelength grids (triggers resampling)
# ---------------------------------------------------------------------------

class TestDiffGrid1D:
    """Spectra have different wavelength arrays, forcing resample."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(1)
        self.x1 = np.linspace(400, 800, 100)
        self.y1 = np.random.rand(100)
        self.x2 = np.linspace(450, 750, 80)
        self.y2 = np.random.rand(80)

    def test_squared_errors(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2)
        assert_equal(old, new, "1D diff grid squared")

    def test_abs_errors(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       squared_errs=False)
        assert_equal(old, new, "1D diff grid abs")

    def test_gradient_squared(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           use_gradient=True)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       use_gradient=True)
        assert_equal(old, new, "1D diff grid gradient squared")

    def test_gradient_abs(self):
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           use_gradient=True, squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       use_gradient=True, squared_errs=False)
        assert_equal(old, new, "1D diff grid gradient abs")

    def test_custom_range(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           custom_range=cr)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       custom_range=cr)
        assert_equal(old, new, "1D diff grid custom_range")

    def test_custom_range_gradient_abs(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.y1, self.x2, self.y2,
                                           custom_range=cr, use_gradient=True,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.y1, self.x2, self.y2,
                                       custom_range=cr, use_gradient=True,
                                       squared_errs=False)
        assert_equal(old, new, "1D diff grid custom_range+gradient+abs")


# ---------------------------------------------------------------------------
# 3D cube tests — same wavelength grid
# ---------------------------------------------------------------------------

class TestSameGridCube:
    """y1 is a 3D cube (rows, cols, bands), same wavelength grid."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(2)
        self.x1 = np.linspace(400, 800, 50)
        self.cube = np.random.rand(8, 12, 50)
        self.x2 = np.linspace(400, 800, 50)
        self.y2 = np.random.rand(50)

    def test_squared_errors(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2)
        assert_equal(old, new, "cube same grid squared")

    def test_abs_errors(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       squared_errs=False)
        assert_equal(old, new, "cube same grid abs")

    def test_gradient_squared(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           use_gradient=True)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       use_gradient=True)
        assert_equal(old, new, "cube same grid gradient squared")

    def test_gradient_abs(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           use_gradient=True, squared_errs=False)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       use_gradient=True, squared_errs=False)
        assert_equal(old, new, "cube same grid gradient abs")

    def test_custom_range(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           custom_range=cr)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       custom_range=cr)
        assert_equal(old, new, "cube same grid custom_range")

    def test_custom_range_gradient_abs(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           custom_range=cr, use_gradient=True,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       custom_range=cr, use_gradient=True,
                                       squared_errs=False)
        assert_equal(old, new, "cube same grid custom_range+gradient+abs")

    def test_output_shape(self):
        result = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2)
        assert result.shape == (8, 12), f"expected (8,12) got {result.shape}"


# ---------------------------------------------------------------------------
# 3D cube tests — different wavelength grids (triggers resampling)
# ---------------------------------------------------------------------------

class TestDiffGridCube:
    """y1 is a 3D cube, different wavelength grid from y2."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(3)
        self.x1 = np.linspace(400, 800, 50)
        self.cube = np.random.rand(8, 12, 50)
        self.x2 = np.linspace(450, 750, 40)
        self.y2 = np.random.rand(40)

    def test_squared_errors(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2)
        assert_equal(old, new, "cube diff grid squared")

    def test_abs_errors(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           squared_errs=False)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       squared_errs=False)
        assert_equal(old, new, "cube diff grid abs")

    def test_gradient_squared(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           use_gradient=True)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       use_gradient=True)
        assert_equal(old, new, "cube diff grid gradient squared")

    def test_gradient_abs(self):
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           use_gradient=True, squared_errs=False)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       use_gradient=True, squared_errs=False)
        assert_equal(old, new, "cube diff grid gradient abs")

    def test_custom_range(self):
        cr = (500, 700)
        old = Database.compare_spectra_old(self.x1, self.cube, self.x2, self.y2,
                                           custom_range=cr)
        new = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2,
                                       custom_range=cr)
        assert_equal(old, new, "cube diff grid custom_range")

    def test_output_shape(self):
        result = Database.compare_spectra(self.x1, self.cube, self.x2, self.y2)
        assert result.shape == (8, 12), f"expected (8,12) got {result.shape}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_identical_spectra_zero_error(self):
        """Comparing a spectrum to itself should yield 0 error."""
        x = np.linspace(400, 800, 100)
        y = np.random.rand(100)
        old = Database.compare_spectra_old(x, y, x, y)
        new = Database.compare_spectra(x, y, x, y)
        assert_equal(old, new, "identical spectra")
        assert new == 0.0

    def test_identical_cube_zero_error(self):
        """Comparing a cube where every pixel equals y2 should give all-zero map."""
        x = np.linspace(400, 800, 50)
        y2 = np.random.rand(50)
        cube = np.broadcast_to(y2, (4, 6, 50)).copy()
        old = Database.compare_spectra_old(x, cube, x, y2)
        new = Database.compare_spectra(x, cube, x, y2)
        assert_equal(old, new, "identical cube")
        np.testing.assert_allclose(new, 0.0, atol=1e-15)

    def test_no_overlap_returns_none(self):
        """Non-overlapping wavelength ranges should return None."""
        x1 = np.linspace(400, 500, 50)
        y1 = np.random.rand(50)
        x2 = np.linspace(600, 700, 50)
        y2 = np.random.rand(50)
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert old is None
        assert new is None

    def test_minimal_overlap(self):
        """Only a few overlapping wavelength points."""
        x1 = np.linspace(400, 500, 50)
        y1 = np.random.rand(50)
        x2 = np.linspace(498, 600, 50)
        y2 = np.random.rand(50)
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old, new, "minimal overlap")

    def test_custom_range_outside_data(self):
        """custom_range that is entirely outside the data wavelengths."""
        x1 = np.linspace(400, 500, 50)
        y1 = np.random.rand(50)
        x2 = np.linspace(400, 500, 50)
        y2 = np.random.rand(50)
        cr = (600, 700)
        old = Database.compare_spectra_old(x1, y1, x2, y2, custom_range=cr)
        new = Database.compare_spectra(x1, y1, x2, y2, custom_range=cr)
        assert old is None and new is None

    def test_custom_range_partial_overlap(self):
        """custom_range that partially overlaps data range."""
        np.random.seed(10)
        x1 = np.linspace(400, 800, 100)
        y1 = np.random.rand(100)
        x2 = np.linspace(400, 800, 100)
        y2 = np.random.rand(100)
        cr = (350, 500)
        old = Database.compare_spectra_old(x1, y1, x2, y2, custom_range=cr)
        new = Database.compare_spectra(x1, y1, x2, y2, custom_range=cr)
        assert_equal(old, new, "custom_range partial overlap")

    def test_single_point_overlap(self):
        """Exactly 1 overlapping wavelength — overlap check behavior."""
        x1 = np.array([400.0, 500.0])
        y1 = np.array([0.1, 0.2])
        x2 = np.array([500.0, 600.0])
        y2 = np.array([0.3, 0.4])
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old, new, "single point overlap")

    def test_reversed_argument_order(self):
        """Swapping which spectrum is x1/y1 vs x2/y2."""
        np.random.seed(20)
        x1 = np.linspace(400, 800, 100)
        y1 = np.random.rand(100)
        x2 = np.linspace(450, 750, 80)
        y2 = np.random.rand(80)
        old_fwd = Database.compare_spectra_old(x1, y1, x2, y2)
        new_fwd = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old_fwd, new_fwd, "forward order")
        old_rev = Database.compare_spectra_old(x2, y2, x1, y1)
        new_rev = Database.compare_spectra(x2, y2, x1, y1)
        assert_equal(old_rev, new_rev, "reversed order")

    def test_large_cube(self):
        """Larger cube to stress-test broadcasting."""
        np.random.seed(30)
        x1 = np.linspace(400, 800, 200)
        cube = np.random.rand(64, 64, 200)
        x2 = np.linspace(420, 780, 150)
        y2 = np.random.rand(150)
        old = Database.compare_spectra_old(x1, cube, x2, y2)
        new = Database.compare_spectra(x1, cube, x2, y2)
        assert_equal(old, new, "large cube")
        assert new.shape == (64, 64)

    def test_short_spectra(self):
        """Very short spectra (few bands)."""
        np.random.seed(40)
        x1 = np.array([400.0, 500.0, 600.0])
        y1 = np.random.rand(3)
        x2 = np.array([400.0, 500.0, 600.0])
        y2 = np.random.rand(3)
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old, new, "short spectra")

    def test_short_spectra_gradient(self):
        """Gradient on very short spectra (3 bands)."""
        np.random.seed(41)
        x1 = np.array([400.0, 500.0, 600.0])
        y1 = np.random.rand(3)
        x2 = np.array([400.0, 500.0, 600.0])
        y2 = np.random.rand(3)
        old = Database.compare_spectra_old(x1, y1, x2, y2, use_gradient=True)
        new = Database.compare_spectra(x1, y1, x2, y2, use_gradient=True)
        assert_equal(old, new, "short spectra gradient")

    def test_float32_inputs(self):
        """Inputs as float32 arrays."""
        np.random.seed(50)
        x1 = np.linspace(400, 800, 100).astype(np.float32)
        y1 = np.random.rand(100).astype(np.float32)
        x2 = np.linspace(400, 800, 100).astype(np.float32)
        y2 = np.random.rand(100).astype(np.float32)
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old, new, "float32 inputs")

    def test_list_inputs(self):
        """Inputs as plain Python lists (not np.arrays)."""
        np.random.seed(60)
        x1 = np.linspace(400, 800, 50).tolist()
        y1 = np.random.rand(50).tolist()
        x2 = np.linspace(400, 800, 50).tolist()
        y2 = np.random.rand(50).tolist()
        old = Database.compare_spectra_old(x1, y1, x2, y2)
        new = Database.compare_spectra(x1, y1, x2, y2)
        assert_equal(old, new, "list inputs")

    def test_cube_1x1(self):
        """1x1 pixel cube — still 3D shape."""
        np.random.seed(70)
        x = np.linspace(400, 800, 50)
        cube = np.random.rand(1, 1, 50)
        y2 = np.random.rand(50)
        old = Database.compare_spectra_old(x, cube, x, y2)
        new = Database.compare_spectra(x, cube, x, y2)
        assert_equal(old, new, "1x1 cube")
        assert new.shape == (1, 1)

    def test_custom_range_tight(self):
        """custom_range that selects only 2-3 bands."""
        np.random.seed(80)
        x = np.linspace(400, 800, 100)
        y1 = np.random.rand(100)
        y2 = np.random.rand(100)
        cr = (400, 408)  # ~2 points at 4.04 nm spacing
        old = Database.compare_spectra_old(x, y1, x, y2, custom_range=cr)
        new = Database.compare_spectra(x, y1, x, y2, custom_range=cr)
        assert_equal(old, new, "tight custom_range")


# ---------------------------------------------------------------------------
# Combinatorial sweep — all flag combos
# ---------------------------------------------------------------------------

class TestCombinatorial:
    """Test every combination of (use_gradient, squared_errs) x (same/diff grid) x (1D/cube)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(99)
        self.x_same = np.linspace(400, 800, 60)
        self.x_diff = np.linspace(420, 780, 45)
        self.y1_1d = np.random.rand(60)
        self.y2_same = np.random.rand(60)
        self.y2_diff = np.random.rand(45)
        self.cube = np.random.rand(5, 7, 60)

    @pytest.mark.parametrize("use_gradient", [False, True])
    @pytest.mark.parametrize("squared_errs", [False, True])
    def test_1d_same_grid(self, use_gradient, squared_errs):
        old = Database.compare_spectra_old(self.x_same, self.y1_1d,
                                           self.x_same, self.y2_same,
                                           use_gradient=use_gradient,
                                           squared_errs=squared_errs)
        new = Database.compare_spectra(self.x_same, self.y1_1d,
                                       self.x_same, self.y2_same,
                                       use_gradient=use_gradient,
                                       squared_errs=squared_errs)
        assert_equal(old, new, f"combo 1d same grad={use_gradient} sq={squared_errs}")

    @pytest.mark.parametrize("use_gradient", [False, True])
    @pytest.mark.parametrize("squared_errs", [False, True])
    def test_1d_diff_grid(self, use_gradient, squared_errs):
        old = Database.compare_spectra_old(self.x_same, self.y1_1d,
                                           self.x_diff, self.y2_diff,
                                           use_gradient=use_gradient,
                                           squared_errs=squared_errs)
        new = Database.compare_spectra(self.x_same, self.y1_1d,
                                       self.x_diff, self.y2_diff,
                                       use_gradient=use_gradient,
                                       squared_errs=squared_errs)
        assert_equal(old, new, f"combo 1d diff grad={use_gradient} sq={squared_errs}")

    @pytest.mark.parametrize("use_gradient", [False, True])
    @pytest.mark.parametrize("squared_errs", [False, True])
    def test_cube_same_grid(self, use_gradient, squared_errs):
        old = Database.compare_spectra_old(self.x_same, self.cube,
                                           self.x_same, self.y2_same,
                                           use_gradient=use_gradient,
                                           squared_errs=squared_errs)
        new = Database.compare_spectra(self.x_same, self.cube,
                                       self.x_same, self.y2_same,
                                       use_gradient=use_gradient,
                                       squared_errs=squared_errs)
        assert_equal(old, new, f"combo cube same grad={use_gradient} sq={squared_errs}")

    @pytest.mark.parametrize("use_gradient", [False, True])
    @pytest.mark.parametrize("squared_errs", [False, True])
    def test_cube_diff_grid(self, use_gradient, squared_errs):
        old = Database.compare_spectra_old(self.x_same, self.cube,
                                           self.x_diff, self.y2_diff,
                                           use_gradient=use_gradient,
                                           squared_errs=squared_errs)
        new = Database.compare_spectra(self.x_same, self.cube,
                                       self.x_diff, self.y2_diff,
                                       use_gradient=use_gradient,
                                       squared_errs=squared_errs)
        assert_equal(old, new, f"combo cube diff grad={use_gradient} sq={squared_errs}")
