import os
import json
import time
import shutil
import numpy as np
from scipy.signal import resample

from hyperlyse.feature_extractor import FeatureExtractor


PIPELINE_VERSION = "1"
CUBE_DATA_EXTENSIONS = {'.raw', '.dat', '.bil'}
REFERENCE_PREFIXES = ('DARKREF_', 'WHITEREF_')


def discover_cubes(folder, include_subfolders=False):
    """
    Scan a folder for hyperspectral cube data files.

    Returns a sorted list of absolute file paths. Excludes .hdr files,
    dark/white reference files, and files in .hyperlyse_cache.
    """
    results = []
    if not folder or not os.path.isdir(folder):
        return results

    if include_subfolders:
        for root, dirs, files in os.walk(folder):
            # Skip cache directories
            dirs[:] = [d for d in dirs if d != '.hyperlyse_cache']
            for f in files:
                if _is_cube_file(f):
                    results.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and _is_cube_file(f):
                results.append(full)

    results.sort()
    return results


def _is_cube_file(filename):
    """Check if a filename is a valid cube data file."""
    base = os.path.basename(filename)
    _, ext = os.path.splitext(base)
    if ext.lower() not in CUBE_DATA_EXTENSIONS:
        return False
    for prefix in REFERENCE_PREFIXES:
        if base.startswith(prefix):
            return False
    return True


def _cache_dir_for_cube(cube_folder, cube_filepath):
    """Return the cache directory path for a given cube file."""
    rel = os.path.relpath(cube_filepath, cube_folder)
    # Sanitize path separators and dots for directory name
    sanitized = rel.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    sanitized = sanitized.replace('.', '_')
    return os.path.join(cube_folder, '.hyperlyse_cache', 'cube_vectors', sanitized)


def _is_cache_valid(cache_dir, cube_filepath, sample_rate):
    """Check if a cube's cache is valid and complete."""
    meta_path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if meta.get('status') != 'complete':
        return False
    if meta.get('pipeline_version') != PIPELINE_VERSION:
        return False
    if meta.get('sample_rate') != sample_rate:
        return False

    # Check file modification
    try:
        stat = os.stat(cube_filepath)
        if meta.get('file_size') != stat.st_size:
            return False
        if meta.get('file_mtime') != stat.st_mtime:
            return False
    except OSError:
        return False

    # Check that data files exist
    if not os.path.isfile(os.path.join(cache_dir, 'spectra.npy')):
        return False
    if not os.path.isfile(os.path.join(cache_dir, 'rgb_preview.npy')):
        return False

    return True


def analyze_cube(cube_filepath, cube_folder, sample_rate=1, cube_class=None):
    """
    Analyze a single cube: load it, extract spectra, cache to disk.

    :param cube_filepath: Path to the cube data file.
    :param cube_folder: Root folder for the cube collection (cache lives here).
    :param sample_rate: Spatial sampling rate (1 = every pixel).
    :param cube_class: The Cube class to use for loading (for dependency injection in tests).
    :return: cache_dir path on success.
    """
    if cube_class is None:
        from hyperlyse.cube import Cube
        cube_class = Cube

    cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)

    # Clean up any partial cache
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Write initial metadata with status "analyzing"
    stat = os.stat(cube_filepath)
    meta = {
        'cube_file': cube_filepath,
        'pipeline_version': PIPELINE_VERSION,
        'file_mtime': stat.st_mtime,
        'file_size': stat.st_size,
        'sample_rate': sample_rate,
        'status': 'analyzing',
        'timestamp': time.time(),
    }
    meta_path = os.path.join(cache_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Load cube
    cube = cube_class(cube_filepath)

    # Extract and sample spectra
    if sample_rate > 1:
        spectra = cube.data[::sample_rate, ::sample_rate, :].astype(np.float32)
    else:
        spectra = cube.data.astype(np.float32)

    # Save spectra
    np.save(os.path.join(cache_dir, 'spectra.npy'), spectra)

    # Save RGB preview
    rgb = cube.to_rgb()
    rgb_uint8 = np.uint8(np.clip(rgb, 0, 1) * 255)
    np.save(os.path.join(cache_dir, 'rgb_preview.npy'), rgb_uint8)

    # Update metadata to complete
    meta['nrows'] = cube.nrows
    meta['ncols'] = cube.ncols
    meta['nbands'] = cube.nbands
    meta['bands'] = list(cube.bands)
    meta['sampled_rows'] = spectra.shape[0]
    meta['sampled_cols'] = spectra.shape[1]
    meta['status'] = 'complete'
    meta['timestamp'] = time.time()
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return cache_dir


def analyze_cubes(cube_folder, sample_rate=1, include_subfolders=False,
                  cube_class=None, progress_callback=None):
    """
    Discover and analyze all cubes in a folder.

    :param cube_folder: Root folder containing cubes.
    :param sample_rate: Spatial sampling rate.
    :param include_subfolders: Whether to recurse into subdirectories.
    :param cube_class: The Cube class (for testing).
    :param progress_callback: Optional callable(current_index, total, cube_name, elapsed_per_cube).
    :return: List of (cube_filepath, cache_dir) tuples for analyzed/cached cubes.
    """
    cube_files = discover_cubes(cube_folder, include_subfolders)
    results = []
    elapsed_times = []

    for i, cube_filepath in enumerate(cube_files):
        cube_name = os.path.basename(cube_filepath)
        cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)

        if _is_cache_valid(cache_dir, cube_filepath, sample_rate):
            results.append((cube_filepath, cache_dir))
            if progress_callback:
                progress_callback(i, len(cube_files), cube_name, 0, skipped=True)
            continue

        t0 = time.time()
        try:
            cache_dir = analyze_cube(cube_filepath, cube_folder, sample_rate, cube_class)
            results.append((cube_filepath, cache_dir))
        except Exception as e:
            print(f"Error analyzing {cube_filepath}: {e}")

        elapsed = time.time() - t0
        elapsed_times.append(elapsed)

        if progress_callback:
            avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
            progress_callback(i, len(cube_files), cube_name, avg_time, skipped=False)

    return results


def search_in_cached_cubes(cube_folder, x_query, y_query,
                           sample_rate=1, include_subfolders=False,
                           custom_range=None, use_gradient=False,
                           squared_errs=True, num_hits=3,
                           progress_callback=None):
    """
    Search across all cached cubes for spectra similar to the query.

    :param cube_folder: Root folder with cached cubes.
    :param x_query: Wavelength array of the query spectrum.
    :param y_query: Intensity array (1D only).
    :param sample_rate: Must match the sample_rate used during analysis.
    :param include_subfolders: Whether subfolders were included in analysis.
    :param custom_range: (x_min, x_max) wavelength range for comparison.
    :param use_gradient: Compare gradients instead of raw spectra.
    :param squared_errs: Use squared errors.
    :param num_hits: Number of top hits to return.
    :param progress_callback: Optional callable(current, total, cube_name, avg_time).
    :return: List of hit dicts sorted by error, up to num_hits entries.
    """
    x_query = np.array(x_query)
    y_query = np.array(y_query)

    if len(y_query.shape) != 1:
        raise ValueError("y_query must be 1D for cube search")

    extractor = FeatureExtractor()
    all_hits = []

    cube_files = discover_cubes(cube_folder, include_subfolders)
    total = len(cube_files)
    t_start = time.time()

    for i, cube_filepath in enumerate(cube_files):
        cube_name = os.path.splitext(os.path.basename(cube_filepath))[0]
        avg_time = (time.time() - t_start) / (i + 1) if i > 0 else 0
        if progress_callback is not None:
            progress_callback(i, total, cube_name, avg_time)
        cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)

        if not _is_cache_valid(cache_dir, cube_filepath, sample_rate):
            continue

        # Load metadata
        with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
            meta = json.load(f)

        x_cube = np.array(meta['bands'])

        # Compute overlapping range
        lambda_min = max(x_query[0], x_cube[0])
        lambda_max = min(x_query[-1], x_cube[-1])
        if custom_range is not None:
            lambda_min = max(lambda_min, custom_range[0])
            lambda_max = min(lambda_max, custom_range[1])

        mask_query = np.logical_and(x_query >= lambda_min, x_query <= lambda_max)
        mask_cube = np.logical_and(x_cube >= lambda_min, x_cube <= lambda_max)

        if mask_query.sum() < 2 or mask_cube.sum() < 2:
            continue

        effective_range = (lambda_min, lambda_max)

        # Compute query vector
        v_query = extractor.extract(x_query, y_query, effective_range, use_gradient)

        # Load cached spectra
        spectra = np.load(os.path.join(cache_dir, 'spectra.npy'))

        # Resample cube spectra to match query grid if grids differ
        x_query_masked = x_query[mask_query]
        x_cube_masked = x_cube[mask_cube]

        if not np.array_equal(x_query_masked, x_cube_masked):
            # Resample each pixel's spectrum from cube grid to query grid
            cube_masked = spectra[:, :, mask_cube]
            nrows, ncols, _ = cube_masked.shape
            n_target = mask_query.sum()
            resampled = np.empty((nrows, ncols, n_target), dtype=np.float32)
            for r in range(nrows):
                for c in range(ncols):
                    resampled[r, c, :] = resample(cube_masked[r, c, :], n_target)
            cube_spectra = resampled
            x_for_extract = x_query_masked
        else:
            cube_spectra = spectra[:, :, mask_cube]
            x_for_extract = x_cube_masked

        # Apply spectrum_to_vector to the entire cube at once (3D)
        v_cube = extractor.extract(x_for_extract, cube_spectra, effective_range, use_gradient)

        # Compute distances: v_cube is (rows, cols, vec_dim), v_query is (vec_dim,)
        errs = v_cube - v_query
        if squared_errs:
            errs = np.power(errs, 2)
        else:
            errs = np.abs(errs)
        error_map = np.mean(errs, axis=2)  # (rows, cols)

        # Find top hits in this cube
        flat_indices = np.argsort(error_map, axis=None)[:num_hits]
        sr = meta['sample_rate']

        for idx in flat_indices:
            sy, sx = np.unravel_index(idx, error_map.shape)
            error_val = error_map[sy, sx]
            # Map back to original cube coordinates
            orig_x = int(sx * sr)
            orig_y = int(sy * sr)
            # Get the spectrum at this location from the cached data
            hit_spectrum = spectra[sy, sx, :]

            all_hits.append({
                'error': float(error_val),
                'cube_file': cube_filepath,
                'cube_name': os.path.splitext(os.path.basename(cube_filepath))[0],
                'x': orig_x,
                'y': orig_y,
                'sampled_x': int(sx),
                'sampled_y': int(sy),
                'spectrum_y': hit_spectrum,
                'spectrum_x': np.array(meta['bands']),
                'cache_dir': cache_dir,
                'nrows': meta['nrows'],
                'ncols': meta['ncols'],
            })

    # Sort all hits by error and return top N
    all_hits.sort(key=lambda h: h['error'])
    return all_hits[:num_hits]


def get_cached_cube_dirs(cube_folder, include_subfolders=False, sample_rate=1):
    """Return list of valid cache directories for all analyzed cubes."""
    cube_files = discover_cubes(cube_folder, include_subfolders)
    result = []
    for cube_filepath in cube_files:
        cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)
        if _is_cache_valid(cache_dir, cube_filepath, sample_rate):
            result.append(cache_dir)
    return result


def reset_cache(cube_folder):
    """Delete all cached cube analysis data."""
    cache_dir = os.path.join(cube_folder, '.hyperlyse_cache', 'cube_vectors')
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def load_rgb_preview(cache_dir):
    """Load the RGB preview image from a cube's cache directory."""
    path = os.path.join(cache_dir, 'rgb_preview.npy')
    if os.path.isfile(path):
        return np.load(path)
    return None
