import sys
import os
import json
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperlyse.cube_analyzer import (
    discover_cubes, analyze_cube, analyze_cubes, _cache_dir_for_cube,
    _is_cache_valid, search_in_cached_cubes, reset_cache, load_rgb_preview,
    PIPELINE_VERSION,
)


# ---------------------------------------------------------------------------
# Mock Cube class for unit tests (avoids needing the spectral library)
# ---------------------------------------------------------------------------

class MockCube:
    """A minimal Cube stand-in that takes a numpy array directly."""

    def __init__(self, filepath):
        """Load synthetic data from a companion .npz file."""
        npz_path = filepath + '.mock.npz'
        if os.path.isfile(npz_path):
            data = np.load(npz_path)
            self.data = data['data']
            self.bands = list(data['bands'])
        else:
            # Default: 4x4 cube with 10 bands
            np.random.seed(42)
            self.data = np.random.rand(4, 4, 10).astype(np.float64)
            self.bands = list(np.linspace(400, 800, 10))
        self.nrows = self.data.shape[0]
        self.ncols = self.data.shape[1]
        self.nbands = self.data.shape[2]
        self.device = 'MockDevice'

    def to_rgb(self):
        # Return a simple 3-channel slice, clipped to [0,1]
        r = min(self.nbands - 1, 0)
        g = min(self.nbands - 1, 1)
        b = min(self.nbands - 1, 2)
        rgb = self.data[:, :, [r, g, b]]
        return np.clip(rgb, 0, 1)


def _create_dummy_cube_file(path):
    """Create a dummy file that looks like a cube data file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\x00' * 64)


def _create_mock_cube_with_data(path, data, bands):
    """Create a dummy cube file with companion mock data."""
    _create_dummy_cube_file(path)
    np.savez(path + '.mock.npz', data=data, bands=bands)


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverCubes:

    def test_discovers_raw_dat_bil(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'cube1.raw'))
        _create_dummy_cube_file(str(tmp_path / 'cube2.dat'))
        _create_dummy_cube_file(str(tmp_path / 'cube3.bil'))
        result = discover_cubes(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert sorted(names) == ['cube1.raw', 'cube2.dat', 'cube3.bil']

    def test_excludes_darkref_whiteref(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'capture.raw'))
        _create_dummy_cube_file(str(tmp_path / 'DARKREF_capture.raw'))
        _create_dummy_cube_file(str(tmp_path / 'WHITEREF_capture.raw'))
        result = discover_cubes(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == ['capture.raw']

    def test_excludes_hdr_files(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'capture.raw'))
        _create_dummy_cube_file(str(tmp_path / 'capture.hdr'))
        result = discover_cubes(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == ['capture.raw']

    def test_no_subfolders_by_default(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'top.raw'))
        sub = tmp_path / 'sub'
        _create_dummy_cube_file(str(sub / 'nested.raw'))
        result = discover_cubes(str(tmp_path), include_subfolders=False)
        names = [os.path.basename(p) for p in result]
        assert names == ['top.raw']

    def test_include_subfolders(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'top.raw'))
        sub = tmp_path / 'sub'
        _create_dummy_cube_file(str(sub / 'nested.raw'))
        result = discover_cubes(str(tmp_path), include_subfolders=True)
        names = sorted([os.path.basename(p) for p in result])
        assert names == ['nested.raw', 'top.raw']

    def test_empty_folder(self, tmp_path):
        result = discover_cubes(str(tmp_path))
        assert result == []

    def test_only_hdr_and_refs(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'capture.hdr'))
        _create_dummy_cube_file(str(tmp_path / 'DARKREF_capture.raw'))
        result = discover_cubes(str(tmp_path))
        assert result == []

    def test_nonexistent_folder(self):
        result = discover_cubes('/nonexistent/path')
        assert result == []

    def test_skips_cache_directory(self, tmp_path):
        _create_dummy_cube_file(str(tmp_path / 'cube.raw'))
        cache_dir = tmp_path / '.hyperlyse_cache' / 'cube_vectors'
        _create_dummy_cube_file(str(cache_dir / 'fake.raw'))
        result = discover_cubes(str(tmp_path), include_subfolders=True)
        names = [os.path.basename(p) for p in result]
        assert names == ['cube.raw']


# ---------------------------------------------------------------------------
# Analysis pipeline tests
# ---------------------------------------------------------------------------

class TestAnalyzeCube:

    def test_produces_cache_structure(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        assert os.path.isfile(os.path.join(cache_dir, 'spectra.npy'))
        assert os.path.isfile(os.path.join(cache_dir, 'metadata.json'))
        assert os.path.isfile(os.path.join(cache_dir, 'rgb_preview.npy'))

    def test_spectra_shape_sample_rate_1(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        spectra = np.load(os.path.join(cache_dir, 'spectra.npy'))
        # MockCube default is 4x4x10
        assert spectra.shape == (4, 4, 10)
        assert spectra.dtype == np.float32

    def test_spectra_shape_sample_rate_2(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=2, cube_class=MockCube)

        spectra = np.load(os.path.join(cache_dir, 'spectra.npy'))
        # 4//2 = 2
        assert spectra.shape == (2, 2, 10)

    def test_metadata_complete(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)

        assert meta['status'] == 'complete'
        assert meta['pipeline_version'] == PIPELINE_VERSION
        assert meta['sample_rate'] == 1
        assert meta['nrows'] == 4
        assert meta['ncols'] == 4
        assert meta['nbands'] == 10
        assert len(meta['bands']) == 10
        assert 'file_mtime' in meta
        assert 'file_size' in meta

    def test_rgb_preview_shape_and_dtype(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        rgb = np.load(os.path.join(cache_dir, 'rgb_preview.npy'))
        assert rgb.shape == (4, 4, 3)
        assert rgb.dtype == np.uint8

    def test_cleans_partial_cache(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)

        # Create a partial cache
        cache_dir = _cache_dir_for_cube(str(tmp_path), cube_file)
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump({'status': 'analyzing'}, f)

        # Analyze should overwrite it
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)
        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        assert meta['status'] == 'complete'


class TestAnalyzeCubes:

    def test_skips_already_analyzed(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)

        # First run
        results1 = analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)
        assert len(results1) == 1

        # Record mtime of cache
        cache_dir = results1[0][1]
        meta_mtime1 = os.path.getmtime(os.path.join(cache_dir, 'metadata.json'))

        # Short sleep to make mtime differ if rewritten
        time.sleep(0.1)

        # Second run — should skip
        skipped = []
        def cb(i, total, name, avg, skipped=False):
            if skipped:
                skipped.append(name) if hasattr(skipped, 'append') else None

        results2 = analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)
        assert len(results2) == 1

        # metadata.json should NOT have been rewritten
        meta_mtime2 = os.path.getmtime(os.path.join(cache_dir, 'metadata.json'))
        assert meta_mtime1 == meta_mtime2

    def test_reanalyzes_when_sample_rate_changed(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)

        analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)
        results = analyze_cubes(str(tmp_path), sample_rate=2, cube_class=MockCube)
        cache_dir = results[0][1]
        spectra = np.load(os.path.join(cache_dir, 'spectra.npy'))
        assert spectra.shape == (2, 2, 10)

    def test_reanalyzes_incomplete_cache(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)

        # Create an incomplete cache manually
        cache_dir = _cache_dir_for_cube(str(tmp_path), cube_file)
        os.makedirs(cache_dir, exist_ok=True)
        stat = os.stat(cube_file)
        meta = {
            'status': 'analyzing',
            'pipeline_version': PIPELINE_VERSION,
            'sample_rate': 1,
            'file_mtime': stat.st_mtime,
            'file_size': stat.st_size,
        }
        with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f)

        results = analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)
        assert len(results) == 1
        with open(os.path.join(results[0][1], 'metadata.json'), 'r') as f:
            meta = json.load(f)
        assert meta['status'] == 'complete'


# ---------------------------------------------------------------------------
# Cache management tests
# ---------------------------------------------------------------------------

class TestCacheManagement:

    def test_reset_cache_deletes_everything(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        cache_root = os.path.join(str(tmp_path), '.hyperlyse_cache', 'cube_vectors')
        assert os.path.isdir(cache_root)

        reset_cache(str(tmp_path))
        assert not os.path.isdir(cache_root)

    def test_reset_cache_nonexistent_is_noop(self, tmp_path):
        # Should not raise
        reset_cache(str(tmp_path))

    def test_load_rgb_preview(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        rgb = load_rgb_preview(cache_dir)
        assert rgb is not None
        assert rgb.shape == (4, 4, 3)
        assert rgb.dtype == np.uint8

    def test_load_rgb_preview_missing(self, tmp_path):
        rgb = load_rgb_preview(str(tmp_path))
        assert rgb is None


# ---------------------------------------------------------------------------
# Cross-cube search tests
# ---------------------------------------------------------------------------

class TestSearchInCachedCubes:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        np.random.seed(0)
        # Create a cube with known data
        self.bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        # Plant a known spectrum at (1,2)
        self.target_spectrum = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = self.target_spectrum
        cube_file = str(tmp_path / 'test_cube.raw')
        _create_mock_cube_with_data(cube_file, data, self.bands)
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)
        self.cube_file = cube_file

    def test_finds_exact_match(self):
        # Query = the planted spectrum
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1)
        assert len(results) >= 1
        best = results[0]
        assert best['error'] < 1e-10  # near-zero error for exact match
        assert best['x'] == 2
        assert best['y'] == 1

    def test_returns_up_to_num_hits(self):
        query = np.random.rand(20)
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=3)
        assert len(results) == 3

    def test_results_sorted_by_error(self):
        query = np.random.rand(20)
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=5)
        errors = [r['error'] for r in results]
        assert errors == sorted(errors)

    def test_result_contains_required_fields(self):
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1)
        hit = results[0]
        assert 'error' in hit
        assert 'cube_file' in hit
        assert 'cube_name' in hit
        assert 'x' in hit
        assert 'y' in hit
        assert 'spectrum_y' in hit
        assert 'spectrum_x' in hit
        assert 'cache_dir' in hit

    def test_empty_when_no_cubes(self, tmp_path):
        empty_dir = str(tmp_path / 'empty')
        os.makedirs(empty_dir)
        results = search_in_cached_cubes(
            empty_dir, self.bands, self.target_spectrum,
            sample_rate=1, num_hits=3)
        assert results == []

    def test_sample_rate_coordinate_mapping(self, tmp_path):
        bands = np.linspace(400, 800, 10)
        data = np.random.rand(8, 8, 10).astype(np.float64)
        # Plant target at (4, 6) in original coords -> (2, 3) in sampled coords
        target = np.ones(10) * 0.5
        data[4, 6, :] = target
        cube_file = str(tmp_path / 'sampled.raw')
        _create_mock_cube_with_data(cube_file, data, bands)
        analyze_cube(cube_file, str(tmp_path), sample_rate=2, cube_class=MockCube)

        results = search_in_cached_cubes(
            str(tmp_path), bands, target,
            sample_rate=2, num_hits=1)
        assert len(results) >= 1
        best = results[0]
        # Sampled (2,3) -> original (4,6)
        assert best['x'] == 6
        assert best['y'] == 4

    def test_multiple_hits_same_cube(self):
        query = np.random.rand(20)
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=3)
        # All hits from the same cube
        cube_files = [r['cube_file'] for r in results]
        assert all(cf == self.cube_file for cf in cube_files)
        # All have distinct coordinates
        coords = [(r['x'], r['y']) for r in results]
        assert len(set(coords)) == len(coords)

    def test_gradient_flag_changes_results(self):
        query = np.random.rand(20)
        r1 = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=1, use_gradient=False)
        r2 = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=1, use_gradient=True)
        # Results should differ (different errors at least)
        assert r1[0]['error'] != r2[0]['error']

    def test_custom_range(self):
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, custom_range=(500, 700))
        assert len(results) >= 1
        # Still finds the match
        assert results[0]['error'] < 1e-10


class TestSearchMultipleCubes:

    def test_merges_results_across_cubes(self, tmp_path):
        np.random.seed(0)
        bands = np.linspace(400, 800, 10)

        # Cube A: target at (0,0)
        data_a = np.random.rand(3, 3, 10).astype(np.float64)
        target = np.linspace(0.2, 0.8, 10)
        data_a[0, 0, :] = target
        cube_a = str(tmp_path / 'cube_a.raw')
        _create_mock_cube_with_data(cube_a, data_a, bands)

        # Cube B: target at (1,1), slightly different
        data_b = np.random.rand(3, 3, 10).astype(np.float64)
        data_b[1, 1, :] = target + 0.01  # close but not exact
        cube_b = str(tmp_path / 'cube_b.raw')
        _create_mock_cube_with_data(cube_b, data_b, bands)

        analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)

        results = search_in_cached_cubes(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=5)

        assert len(results) == 5
        # Best hit should be the exact match in cube_a
        assert results[0]['cube_file'] == cube_a
        assert results[0]['x'] == 0
        assert results[0]['y'] == 0
        assert results[0]['error'] < 1e-10

        # Errors should be sorted
        errors = [r['error'] for r in results]
        assert errors == sorted(errors)

    def test_search_progress_callback(self, tmp_path):
        bands = np.linspace(400, 800, 10)
        data = np.random.rand(3, 3, 10).astype(np.float64)
        target = np.linspace(0.2, 0.8, 10)

        for name in ['cube_x.raw', 'cube_y.raw']:
            _create_mock_cube_with_data(str(tmp_path / name), data, bands)

        analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)

        progress_calls = []
        def on_progress(current, total, cube_name, avg_time):
            progress_calls.append((current, total, cube_name, avg_time))

        results = search_in_cached_cubes(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=3,
            progress_callback=on_progress)

        # Should get one callback per cube
        assert len(progress_calls) == 2
        # total should be 2 in each call
        assert all(call[1] == 2 for call in progress_calls)
        # current should be 0 and 1
        assert progress_calls[0][0] == 0
        assert progress_calls[1][0] == 1
        # cube names should be non-empty strings
        assert all(len(call[2]) > 0 for call in progress_calls)
