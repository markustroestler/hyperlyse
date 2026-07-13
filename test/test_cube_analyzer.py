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

    def test_prefers_reflectance_over_raw_specim_layout(self, tmp_path):
        """Specim IQ scene: capture/<id>.raw + results/REFLECTANCE_<id>.dat.
        Only the reflectance product should be returned."""
        scene = tmp_path / 'Cod_22_f1r_2026-04-17_026'
        _create_dummy_cube_file(str(scene / 'capture' / '2026-04-17_026.raw'))
        _create_dummy_cube_file(str(scene / 'capture' / 'DARKREF_2026-04-17_026.raw'))
        _create_dummy_cube_file(str(scene / 'capture' / 'WHITEREF_2026-04-17_026.raw'))
        _create_dummy_cube_file(str(scene / 'results' / 'REFLECTANCE_2026-04-17_026.dat'))
        result = discover_cubes(str(tmp_path), include_subfolders=True)
        names = [os.path.basename(p) for p in result]
        assert names == ['REFLECTANCE_2026-04-17_026.dat']

    def test_falls_back_to_raw_when_no_reflectance(self, tmp_path):
        """A scene with only the raw capture (no reflectance) keeps the raw."""
        scene = tmp_path / 'Cod_22_f1r_2026-04-17_026'
        _create_dummy_cube_file(str(scene / 'capture' / '2026-04-17_026.raw'))
        _create_dummy_cube_file(str(scene / 'capture' / 'DARKREF_2026-04-17_026.raw'))
        result = discover_cubes(str(tmp_path), include_subfolders=True)
        names = [os.path.basename(p) for p in result]
        assert names == ['2026-04-17_026.raw']

    def test_prefers_reflectance_same_folder(self, tmp_path):
        """Even in a flat folder, REFLECTANCE_<id> wins over <id>.raw."""
        _create_dummy_cube_file(str(tmp_path / 'capture.raw'))
        _create_dummy_cube_file(str(tmp_path / 'REFLECTANCE_capture.dat'))
        result = discover_cubes(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == ['REFLECTANCE_capture.dat']

    def test_distinct_captures_not_collapsed(self, tmp_path):
        """Different capture ids under the same scene root are kept separate."""
        scene = tmp_path / 'scene'
        _create_dummy_cube_file(str(scene / 'capture' / 'cap_001.raw'))
        _create_dummy_cube_file(str(scene / 'capture' / 'cap_002.raw'))
        result = discover_cubes(str(tmp_path), include_subfolders=True)
        names = sorted(os.path.basename(p) for p in result)
        assert names == ['cap_001.raw', 'cap_002.raw']


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

    def test_exclude_cube_file_skips_that_cube(self):
        # Excluding the only cube leaves no cubes to search.
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, exclude_cube_file=self.cube_file)
        assert results == []

    def test_exclude_cube_file_none_searches_all(self):
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, exclude_cube_file=None)
        assert len(results) >= 1

    def test_exclude_other_cube_keeps_match(self, tmp_path):
        # Excluding an unrelated path must not drop the real hit.
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1,
            exclude_cube_file=str(tmp_path / 'some_other_cube.raw'))
        assert len(results) >= 1
        assert results[0]['cube_file'] == self.cube_file

    def test_exclude_raw_capture_skips_reflectance_scene(self, tmp_path):
        # Specim layout: a scene has both capture/<id>.raw and
        # results/REFLECTANCE_<id>.dat. discover_cubes returns only the
        # reflectance .dat; excluding the raw capture the user opened must
        # still skip that scene's cached reflectance product.
        scene = tmp_path / 'scenes' / 'Cod_2026-04-17_026'
        bands = np.linspace(400, 800, 12)
        data = np.random.rand(4, 4, 12).astype(np.float64)
        target = np.linspace(0.2, 0.8, 12)
        data[0, 0, :] = target

        raw_file = str(scene / 'capture' / '2026-04-17_026.raw')
        dat_file = str(scene / 'results' / 'REFLECTANCE_2026-04-17_026.dat')
        _create_mock_cube_with_data(raw_file, data, bands)
        _create_mock_cube_with_data(dat_file, data, bands)

        cube_folder = str(tmp_path / 'scenes')
        analyze_cubes(cube_folder, sample_rate=1, include_subfolders=True,
                      cube_class=MockCube)

        # Only the reflectance .dat is discovered for this scene.
        discovered = discover_cubes(cube_folder, include_subfolders=True)
        assert [os.path.basename(p) for p in discovered] == \
            ['REFLECTANCE_2026-04-17_026.dat']

        # Excluding the raw capture the user opened removes the only scene.
        results = search_in_cached_cubes(
            cube_folder, bands, target, sample_rate=1, num_hits=1,
            include_subfolders=True, exclude_cube_file=raw_file)
        assert results == []

    def test_returns_up_to_num_hits(self, tmp_path):
        # Create 4 additional cubes so there are 5 total.
        for i in range(4):
            data = np.random.rand(4, 4, 20).astype(np.float64)
            cf = str(tmp_path / f'extra_{i}.raw')
            _create_mock_cube_with_data(cf, data, self.bands)
            analyze_cube(cf, str(self.tmp_path), sample_rate=1, cube_class=MockCube)

        query = np.random.rand(20)
        # Ask for 3 cubes from 5 available — should get exactly 3 (one hit each).
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=3)
        assert len(results) == 3
        # Each result comes from a different cube.
        assert len({r['cube_file'] for r in results}) == 3

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


class TestParallelism:
    """Analysis and search fan out over cubes; results must be independent of
    worker count and order-stable."""

    def _make_collection(self, tmp_path, n=6):
        np.random.seed(7)
        bands = np.linspace(400, 800, 12)
        target = np.linspace(0.2, 0.8, 12)
        for k in range(n):
            data = np.random.rand(4, 4, 12).astype(np.float64)
            # Plant the target (with a per-cube offset) so errors are distinct.
            data[k % 4, (k + 1) % 4, :] = target + k * 0.005
            _create_mock_cube_with_data(
                str(tmp_path / f'cube_{k}.raw'), data, bands)
        return bands, target

    def test_analyze_cubes_parallel_matches_sequential(self, tmp_path):
        bands, _ = self._make_collection(tmp_path)
        seq = analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube,
                            max_workers=1)
        # Re-analyze in parallel after a reset to compare returned ordering.
        reset_cache(str(tmp_path))
        par = analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube,
                            max_workers=4)
        # Discovery order is preserved regardless of worker count.
        assert [os.path.basename(c) for c, _ in seq] == \
               [os.path.basename(c) for c, _ in par]
        assert len(par) == 6

    def test_analyze_progress_one_call_per_cube(self, tmp_path):
        self._make_collection(tmp_path)
        calls = []
        def cb(current, total, name, avg, skipped=False):
            calls.append((current, total, name, skipped))
        analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube,
                      progress_callback=cb, max_workers=4)
        assert len(calls) == 6
        assert all(c[1] == 6 for c in calls)
        # Completion indices are 0..5 regardless of finishing order.
        assert sorted(c[0] for c in calls) == [0, 1, 2, 3, 4, 5]

    def test_search_parallel_matches_sequential(self, tmp_path):
        bands, target = self._make_collection(tmp_path)
        analyze_cubes(str(tmp_path), sample_rate=1, cube_class=MockCube)
        seq = search_in_cached_cubes(
            str(tmp_path), bands, target, sample_rate=1, num_hits=5,
            max_workers=1)
        par = search_in_cached_cubes(
            str(tmp_path), bands, target, sample_rate=1, num_hits=5,
            max_workers=4)
        assert [(h['cube_file'], h['x'], h['y']) for h in seq] == \
               [(h['cube_file'], h['x'], h['y']) for h in par]
        assert [h['error'] for h in par] == sorted(h['error'] for h in par)


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

        # num_hits=5 cubes requested, but only 2 cubes exist → 2 results (1 per cube)
        results = search_in_cached_cubes(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=5)

        assert len(results) == 2
        # Each result is from a different cube
        assert len({r['cube_file'] for r in results}) == 2
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


# ---------------------------------------------------------------------------
# Vectorized resampling tests
# ---------------------------------------------------------------------------

class TestSearchMismatchedGrids:

    def test_search_with_different_wavelength_grids(self, tmp_path):
        """When query and cube have different wavelength grids, the vectorized
        resampling path should still find the planted target."""
        np.random.seed(42)
        bands_cube = np.linspace(400, 800, 20)
        bands_query = np.linspace(420, 780, 15)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        # Plant a distinctive target
        target_in_cube = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = target_in_cube
        cube_file = str(tmp_path / 'mismatch.raw')
        _create_mock_cube_with_data(cube_file, data, bands_cube)
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        # Build a query on the different grid — resample the target
        from scipy.signal import resample
        overlap_mask = np.logical_and(bands_cube >= 420, bands_cube <= 780)
        target_overlap = target_in_cube[overlap_mask]
        target_resampled = resample(target_overlap, 15)

        results = search_in_cached_cubes(
            str(tmp_path), bands_query, target_resampled,
            sample_rate=1, num_hits=1)
        assert len(results) >= 1
        assert results[0]['y'] == 1
        assert results[0]['x'] == 2


# ---------------------------------------------------------------------------
# PCA artifact tests
# ---------------------------------------------------------------------------

class TestPCAArtifacts:

    def test_pca_artifacts_produced(self, tmp_path):
        """analyze_cube should produce PCA model and BallTree for fast search."""
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1,
                                 cube_class=MockCube)

        assert os.path.isfile(os.path.join(cache_dir, 'pca_model.joblib'))
        assert os.path.isfile(os.path.join(cache_dir, 'pca_features.npy'))

    def test_pca_metadata_fields(self, tmp_path):
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1,
                                 cube_class=MockCube)

        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        assert 'pca_components' in meta
        assert 'pca_explained_variance_raw' in meta
        assert 'pca_explained_variance_gradient' in meta
        # MockCube has 10 bands, so max 10 components
        assert meta['pca_components'] <= 10
        assert 0 < meta['pca_explained_variance_raw'] <= 1.0
        assert 0 < meta['pca_explained_variance_gradient'] <= 1.0

    def test_pca_spectra_shape(self, tmp_path):
        """Test that PCA model is properly trained on extracted features."""
        cube_file = str(tmp_path / 'capture.raw')
        _create_dummy_cube_file(cube_file)
        cache_dir = analyze_cube(cube_file, str(tmp_path), sample_rate=1,
                                 cube_class=MockCube)

        # Load PCA model and verify it has the right number of components
        import joblib
        pca = joblib.load(os.path.join(cache_dir, 'pca_model.joblib'))
        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        n_components = meta['pca_components']
        assert pca.n_components_ == n_components
        assert pca.explained_variance_ratio_.shape[0] == n_components


# ---------------------------------------------------------------------------
# PCA search tests
# ---------------------------------------------------------------------------

class TestPCASearch:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        np.random.seed(0)
        self.bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        self.target_spectrum = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = self.target_spectrum
        cube_file = str(tmp_path / 'test_cube.raw')
        _create_mock_cube_with_data(cube_file, data, self.bands)
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

    def test_pca_search_finds_exact_match(self):
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, use_pca=True)
        assert len(results) >= 1
        best = results[0]
        assert best['x'] == 2
        assert best['y'] == 1

    def test_pca_search_returns_requested_hits(self):
        # num_hits=3 cubes requested; only 1 cube in the fixture → 1 result
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=3, use_pca=True)
        assert len(results) == 1

    def test_pca_search_has_required_fields(self):
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, use_pca=True)
        hit = results[0]
        for field in ['error', 'cube_file', 'cube_name', 'x', 'y',
                       'spectrum_y', 'spectrum_x', 'cache_dir',
                       'nrows', 'ncols']:
            assert field in hit, f"Missing field: {field}"

    def test_pca_and_bruteforce_agree_on_exact_match(self):
        """Both modes should find the same planted target as top-1."""
        bf = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, use_pca=False)
        pca = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, use_pca=True)
        assert bf[0]['x'] == pca[0]['x']
        assert bf[0]['y'] == pca[0]['y']

    def test_pca_error_comparable_to_bruteforce(self):
        """PCA hits are re-ranked with the exact metric, so the reported error is
        on the same scale as brute-force (not a PCA-space distance)."""
        query = np.random.rand(20)
        bf = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=1, use_pca=False)
        pca = search_in_cached_cubes(
            str(self.tmp_path), self.bands, query,
            sample_rate=1, num_hits=1, use_pca=True)
        # Same winning pixel and the same (comparable) error magnitude.
        assert bf[0]['x'] == pca[0]['x']
        assert bf[0]['y'] == pca[0]['y']
        assert np.isclose(bf[0]['error'], pca[0]['error'])

    def test_pca_respects_custom_range(self):
        """PCA mode honors custom_range when re-ranking candidates."""
        results = search_in_cached_cubes(
            str(self.tmp_path), self.bands, self.target_spectrum,
            sample_rate=1, num_hits=1, use_pca=True, custom_range=(500, 700))
        assert len(results) >= 1
        assert results[0]['x'] == 2
        assert results[0]['y'] == 1
        assert results[0]['error'] < 1e-10

    def test_pca_fallback_when_no_pca_files(self, tmp_path):
        """If PCA artifacts are missing, use_pca=True falls back to brute-force."""
        bands = np.linspace(400, 800, 10)
        data = np.random.rand(4, 4, 10).astype(np.float64)
        target = np.ones(10) * 0.5
        data[2, 2, :] = target
        cube_file = str(tmp_path / 'nopca.raw')
        _create_mock_cube_with_data(cube_file, data, bands)
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        # Delete PCA files to simulate an old cache
        cache_dir = _cache_dir_for_cube(str(tmp_path), cube_file)
        for f in ['pca_model.joblib', 'pca_features.npy', 'pca_features_gradient.npy']:
            path = os.path.join(cache_dir, f)
            if os.path.isfile(path):
                os.remove(path)

        # Search should still work via brute-force fallback
        results = search_in_cached_cubes(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=1, use_pca=True)
        assert len(results) >= 1
        assert results[0]['x'] == 2
        assert results[0]['y'] == 2
