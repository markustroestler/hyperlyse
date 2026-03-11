import inspect
import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hyperlyse.benchmark import (
    timed_operation, benchmark_cube_load, benchmark_analyze_cube,
    benchmark_search, save_benchmark_results, load_benchmark_results,
)


# Reuse MockCube from test_cube_analyzer
from test_cube_analyzer import MockCube, _create_dummy_cube_file, _create_mock_cube_with_data

# Detect whether current production code supports use_pca
from hyperlyse.cube_analyzer import search_in_cached_cubes as _search_fn
_HAS_PCA = 'use_pca' in inspect.signature(_search_fn).parameters


class TestTimedOperation:

    def test_returns_non_negative_elapsed(self):
        results = {}
        with timed_operation("test_op", results):
            _ = sum(range(1000))
        assert "test_op" in results
        assert results["test_op"] >= 0

    def test_works_without_results_dict(self):
        with timed_operation("no_dict"):
            pass

    def test_multiple_operations(self):
        results = {}
        with timed_operation("op1", results):
            pass
        with timed_operation("op2", results):
            pass
        assert "op1" in results
        assert "op2" in results


class TestBenchmarkCubeLoad:

    def test_returns_timing_and_metadata(self, tmp_path):
        cube_file = str(tmp_path / 'cube.raw')
        _create_dummy_cube_file(cube_file)
        results, cube = benchmark_cube_load(cube_file, cube_class=MockCube)
        assert "cube_load_total" in results
        assert results["cube_load_total"] >= 0
        assert results["nrows"] == 4
        assert results["ncols"] == 4
        assert results["nbands"] == 10
        assert results["data_size_mb"] >= 0
        assert cube is not None


class TestBenchmarkAnalyzeCube:

    def test_calls_production_analyze_cube(self, tmp_path):
        """Verify benchmark calls actual analyze_cube and reports results."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        results = benchmark_analyze_cube(
            cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        assert "analyze_total" in results
        assert results["analyze_total"] >= 0
        assert "cache_dir" in results
        assert os.path.isdir(results["cache_dir"])

    @pytest.mark.skipif(not _HAS_PCA,
                        reason="Production code does not yet support PCA")
    def test_reports_pca_and_balltree(self, tmp_path):
        """When production code builds PCA + BallTree, benchmark reports them."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        results = benchmark_analyze_cube(
            cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        assert results["pca_built"] is True
        assert results["balltree_built"] is True
        assert "pca_components" in results

    def test_with_sample_rate(self, tmp_path):
        cube_file = str(tmp_path / 'cube.raw')
        _create_dummy_cube_file(cube_file)
        results = benchmark_analyze_cube(
            cube_file, str(tmp_path), sample_rate=2, cube_class=MockCube)
        assert results["sampled_shape"] == [2, 2, 10]


class TestBenchmarkSearch:

    def test_default_search(self, tmp_path):
        """Verify benchmark always runs the default (brute-force) search."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        target = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = target
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        from hyperlyse.cube_analyzer import analyze_cube
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        results = benchmark_search(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=3)

        # Default search results are always present
        assert "search_total" in results
        assert results["search_total"] >= 0
        assert results["num_hits"] == 3
        assert "best_error" in results
        assert "best_pos" in results

    @pytest.mark.skipif(not _HAS_PCA,
                        reason="Production code does not yet support use_pca")
    def test_pca_search_when_supported(self, tmp_path):
        """When production code has use_pca, benchmark also runs PCA search."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        target = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = target
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        from hyperlyse.cube_analyzer import analyze_cube
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        results = benchmark_search(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=3)

        assert "search_pca_total" in results
        assert results["search_pca_total"] >= 0
        assert results["pca_num_hits"] == 3
        assert "top1_agree" in results

    @pytest.mark.skipif(not _HAS_PCA,
                        reason="Production code does not yet support use_pca")
    def test_exact_match_both_modes_agree(self, tmp_path):
        """When PCA supported, exact match should agree across both modes."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        target = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = target
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        from hyperlyse.cube_analyzer import analyze_cube
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        results = benchmark_search(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=1)

        assert results["top1_agree"] is True

    def test_finds_exact_match(self, tmp_path):
        """Default search finds an exact planted match."""
        bands = np.linspace(400, 800, 20)
        data = np.random.rand(4, 4, 20).astype(np.float64)
        target = np.linspace(0.1, 0.9, 20)
        data[1, 2, :] = target
        cube_file = str(tmp_path / 'cube.raw')
        _create_mock_cube_with_data(cube_file, data, bands)

        from hyperlyse.cube_analyzer import analyze_cube
        analyze_cube(cube_file, str(tmp_path), sample_rate=1, cube_class=MockCube)

        results = benchmark_search(
            str(tmp_path), bands, target,
            sample_rate=1, num_hits=1)

        assert results["num_hits"] >= 1
        assert results["best_error"] < 1e-6


class TestSaveLoadResults:

    def test_roundtrip(self, tmp_path):
        filepath = str(tmp_path / 'bench.jsonl')
        save_benchmark_results({"op": "load", "time": 1.5}, filepath)
        save_benchmark_results({"op": "search", "time": 0.3}, filepath)

        entries = load_benchmark_results(filepath)
        assert len(entries) == 2
        assert entries[0]["op"] == "load"
        assert entries[1]["op"] == "search"
        assert "timestamp" in entries[0]
