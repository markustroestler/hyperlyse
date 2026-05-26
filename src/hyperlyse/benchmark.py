"""Benchmark tool for measuring hyperlyse performance.

Calls the actual production functions so measurements reflect real app behavior.

Usage:
    python -m hyperlyse.benchmark --cube-file path/to/cube.raw --cube-folder path/to/folder
    python -m hyperlyse.benchmark --cube-file path/to/cube.raw --cube-folder path/to/folder --sample-rate 4
    python -m hyperlyse.benchmark --compare --cube-folder path/to/folder
"""
import argparse
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np


RESULTS_FILENAME = 'benchmark_results.jsonl'


@contextmanager
def timed_operation(label, results_dict=None):
    """Context manager that records wall-clock time for an operation."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if results_dict is not None:
        results_dict[label] = round(elapsed, 4)


def benchmark_cube_load(cube_filepath, cube_class=None):
    """Benchmark loading a single cube via the production Cube class.

    Times: cube_load_total (the full Cube() constructor).
    """
    if cube_class is None:
        from hyperlyse.cube import Cube
        cube_class = Cube

    results = {}
    with timed_operation('cube_load_total', results):
        cube = cube_class(cube_filepath)

    results['nrows'] = cube.nrows
    results['ncols'] = cube.ncols
    results['nbands'] = cube.nbands
    results['data_size_mb'] = round(cube.data.nbytes / (1024 * 1024), 1)
    return results, cube


def benchmark_analyze_cube(cube_filepath, cube_folder, sample_rate=1,
                           cube_class=None):
    """Benchmark the production analyze_cube() pipeline.

    Calls the real analyze_cube() function and times the whole thing.
    Also invalidates existing cache first so it measures from scratch.

    Times:
        analyze_total - end-to-end analyze_cube() call
    """
    from hyperlyse.cube_analyzer import analyze_cube, _cache_dir_for_cube
    import shutil

    # Invalidate existing cache so we measure a fresh analysis
    cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    results = {}
    with timed_operation('analyze_total', results):
        cache_dir = analyze_cube(cube_filepath, cube_folder, sample_rate,
                                 cube_class)

    # Report what was produced
    spectra = np.load(os.path.join(cache_dir, 'spectra.npy'))
    results['sampled_shape'] = list(spectra.shape)
    results['spectra_size_mb'] = round(spectra.nbytes / (1024 * 1024), 1)

    # Report PCA/BallTree if they were built (only on new optimized code)
    pca_path = os.path.join(cache_dir, 'pca_model.joblib')
    index_path = os.path.join(cache_dir, 'search_index.joblib')
    if os.path.isfile(pca_path):
        results['pca_built'] = True
    if os.path.isfile(index_path):
        results['balltree_built'] = True

    with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    if 'pca_components' in meta:
        results['pca_components'] = meta['pca_components']

    results['cache_dir'] = cache_dir
    return results


def benchmark_search(cube_folder, x_query, y_query, sample_rate=1,
                     include_subfolders=False, **kwargs):
    """Benchmark the production search_in_cached_cubes() function.

    Always runs the default search. If the production function supports
    use_pca, also runs a PCA+BallTree search for comparison.

    Times:
        search_total     - default search (brute-force or whatever production uses)
        search_pca_total - PCA+BallTree search (if supported by production code)
    """
    import inspect
    from hyperlyse.cube_analyzer import search_in_cached_cubes

    num_hits = kwargs.get('num_hits', 3)
    custom_range = kwargs.get('custom_range', None)
    use_gradient = kwargs.get('use_gradient', False)
    squared_errs = kwargs.get('squared_errs', True)

    # Check if production code supports use_pca parameter
    sig = inspect.signature(search_in_cached_cubes)
    has_pca = 'use_pca' in sig.parameters

    search_kwargs = dict(
        sample_rate=sample_rate,
        include_subfolders=include_subfolders,
        custom_range=custom_range,
        use_gradient=use_gradient,
        squared_errs=squared_errs,
        num_hits=num_hits,
    )

    results = {}

    # Default search (brute-force on old code, still brute-force by default on new)
    with timed_operation('search_total', results):
        hits = search_in_cached_cubes(
            cube_folder, x_query, y_query, **search_kwargs)

    results['num_hits'] = len(hits)
    if hits:
        results['best_error'] = hits[0]['error']
        results['best_pos'] = f"({hits[0]['x']}, {hits[0]['y']})"

    # PCA search (only if production code supports it)
    if has_pca:
        with timed_operation('search_pca_total', results):
            pca_hits = search_in_cached_cubes(
                cube_folder, x_query, y_query,
                use_pca=True, **search_kwargs)

        results['pca_num_hits'] = len(pca_hits)
        if pca_hits:
            results['pca_best_error'] = pca_hits[0]['error']
            results['pca_best_pos'] = f"({pca_hits[0]['x']}, {pca_hits[0]['y']})"

        # Compare: do both modes agree on top-1?
        if hits and pca_hits:
            results['top1_agree'] = (hits[0]['x'] == pca_hits[0]['x'] and
                                     hits[0]['y'] == pca_hits[0]['y'])

    return results


def save_benchmark_results(results, filepath):
    """Append benchmark results with timestamp to a JSON lines file."""
    entry = {'timestamp': time.time(), **results}
    with open(filepath, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_benchmark_results(filepath):
    """Load all benchmark entries from a JSON lines file."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _get_git_commit():
    """Get short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def _print_results(label, results):
    """Print a results dict as a formatted table."""
    print(f'\n=== {label} ===')
    timing_keys = [k for k in results if k.endswith('_total')]
    meta_keys = [k for k in results if k not in timing_keys]

    if timing_keys:
        max_key_len = max(len(k) for k in timing_keys)
        for k in timing_keys:
            v = results[k]
            bar = '#' * int(v * 10)  # rough visual bar, 1 char per 0.1s
            print(f'  {k:<{max_key_len}}  {v:>8.3f}s  {bar}')

    for k in meta_keys:
        v = results[k]
        if k == 'cache_dir':
            continue
        print(f'  {k}: {v}')


def _print_comparison(filepath):
    """Print a comparison table of all saved benchmark runs."""
    entries = load_benchmark_results(filepath)
    if not entries:
        print(f'No benchmark results found in {filepath}')
        return

    print(f'\n=== Benchmark Comparison ({len(entries)} runs) ===\n')

    timing_keys = set()
    for e in entries:
        for k, v in e.items():
            if isinstance(v, (int, float)) and k not in ('timestamp',):
                timing_keys.add(k)

    headers = []
    for e in entries:
        ts = datetime.fromtimestamp(e.get('timestamp', 0))
        commit = e.get('git_commit', '?')
        label = e.get('label', '')
        header = f'{ts:%Y-%m-%d %H:%M} [{commit}]'
        if label:
            header += f' {label}'
        headers.append(header)

    col_width = max(len(h) for h in headers) + 2
    key_width = max(len(k) for k in timing_keys) + 2

    print(f'{"metric":<{key_width}}', end='')
    for h in headers:
        print(f'{h:>{col_width}}', end='')
    print()
    print('-' * (key_width + col_width * len(headers)))

    for k in sorted(timing_keys):
        print(f'{k:<{key_width}}', end='')
        for e in entries:
            v = e.get(k, '')
            if isinstance(v, float):
                print(f'{v:>{col_width}.3f}', end='')
            else:
                print(f'{str(v):>{col_width}}', end='')
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark hyperlyse cube loading, analysis, and search.')
    parser.add_argument('--cube-file',
                        help='Path to a cube data file (.raw/.dat/.bil)')
    parser.add_argument('--cube-folder',
                        help='Path to the folder containing cubes')
    parser.add_argument('--sample-rate', type=int, default=1,
                        help='Spatial sampling rate (default: 1)')
    parser.add_argument('--label', default='',
                        help='Optional label for this benchmark run')
    parser.add_argument('--compare', action='store_true',
                        help='Print comparison of all saved benchmark runs')
    parser.add_argument('--skip-load', action='store_true',
                        help='Skip the cube loading benchmark')
    parser.add_argument('--skip-analyze', action='store_true',
                        help='Skip the analyze benchmark')
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip the search benchmark')
    args = parser.parse_args()

    if args.compare:
        if not args.cube_folder:
            parser.error('--cube-folder is required with --compare')
        results_path = os.path.join(args.cube_folder, RESULTS_FILENAME)
        _print_comparison(results_path)
        return

    if not args.cube_file:
        parser.error('--cube-file is required (unless using --compare)')
    if not args.cube_folder:
        args.cube_folder = os.path.dirname(os.path.abspath(args.cube_file))

    cube_filepath = os.path.abspath(args.cube_file)
    cube_folder = os.path.abspath(args.cube_folder)

    if not os.path.isfile(cube_filepath):
        parser.error(f'Cube file not found: {cube_filepath}')

    file_size_mb = os.path.getsize(cube_filepath) / (1024 * 1024)
    git_commit = _get_git_commit()

    all_results = {
        'git_commit': git_commit,
        'cube_file': os.path.basename(cube_filepath),
        'file_size_mb': round(file_size_mb, 1),
        'sample_rate': args.sample_rate,
        'label': args.label,
    }

    print(f'Benchmarking: {os.path.basename(cube_filepath)} '
          f'({file_size_mb:.0f} MB, sample_rate={args.sample_rate})')
    print(f'Git commit: {git_commit}')

    cube = None

    # 1. Cube loading benchmark
    if not args.skip_load:
        print('\n--- Cube Loading ---')
        load_results, cube = benchmark_cube_load(cube_filepath)
        _print_results('Cube Load', load_results)
        all_results.update(load_results)

    # 2. Analyze benchmark (calls production analyze_cube)
    if not args.skip_analyze:
        print('\n--- Cube Analysis (production analyze_cube) ---')
        analyze_results = benchmark_analyze_cube(
            cube_filepath, cube_folder, args.sample_rate)
        _print_results('Analyze', analyze_results)
        all_results.update({k: v for k, v in analyze_results.items()
                           if k != 'cache_dir'})

    # 3. Search benchmark (uses center pixel as query)
    if not args.skip_search:
        print('\n--- Cube Search ---')
        if cube is None:
            from hyperlyse.cube import Cube
            cube = Cube(cube_filepath)

        cy, cx = cube.nrows // 2, cube.ncols // 2
        x_query = np.array(cube.bands)
        y_query = cube.data[cy, cx, :].flatten().astype(np.float64)

        print(f'Query: center pixel ({cx}, {cy}), {len(x_query)} bands')

        search_results = benchmark_search(
            cube_folder, x_query, y_query,
            sample_rate=args.sample_rate, num_hits=3)
        _print_results('Search', search_results)
        all_results.update(search_results)

    # Save results
    results_path = os.path.join(cube_folder, RESULTS_FILENAME)
    save_benchmark_results(all_results, results_path)
    print(f'\nResults saved to: {results_path}')
    print(f'Run with --compare --cube-folder "{cube_folder}" to compare runs.')


if __name__ == '__main__':
    main()
