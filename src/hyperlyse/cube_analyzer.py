import os
import json
import time
import shutil
import numpy as np
import joblib
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree

from hyperlyse.feature_extractor import FeatureExtractor
from hyperlyse.database import spectrum_to_vector


PIPELINE_VERSION = "2"
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
        from hyperlyse.cube import CubeLazy
        cube_class = CubeLazy

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
    if hasattr(cube, 'get_sampled'):
        spectra = cube.get_sampled(sample_rate)
    else:
        if sample_rate > 1:
            spectra = cube.data[::sample_rate, ::sample_rate, :].astype(np.float32)
        else:
            spectra = cube.data.astype(np.float32)

    # Save spectra
    np.save(os.path.join(cache_dir, 'spectra.npy'), spectra)

    # Extract features and build PCA models for fast search (both raw and gradient modes)
    try:
        n_pixels = spectra.shape[0] * spectra.shape[1]
        n_bands = spectra.shape[2]
        n_components = min(20, n_bands, n_pixels)

        if n_components >= 2 and n_pixels >= n_components:
            x_bands = np.array(cube.bands)

            # Train PCA on RAW features (use_gradient=False)
            features_raw = spectrum_to_vector(x_bands, spectra, custom_range=None, use_gradient=False)
            flat_features_raw = features_raw.reshape(n_pixels, -1)
            pca_raw = PCA(n_components=n_components, svd_solver='auto')
            features_pca_raw = pca_raw.fit_transform(flat_features_raw)

            joblib.dump(pca_raw, os.path.join(cache_dir, 'pca_model.joblib'))
            tree_raw = BallTree(features_pca_raw, metric='euclidean')
            joblib.dump(tree_raw, os.path.join(cache_dir, 'search_index.joblib'))

            meta['pca_components'] = n_components
            meta['pca_explained_variance_raw'] = float(
                np.sum(pca_raw.explained_variance_ratio_))

            # Train PCA on GRADIENT features (use_gradient=True)
            features_grad = spectrum_to_vector(x_bands, spectra, custom_range=None, use_gradient=True)
            flat_features_grad = features_grad.reshape(n_pixels, -1)
            pca_grad = PCA(n_components=n_components, svd_solver='auto')
            features_pca_grad = pca_grad.fit_transform(flat_features_grad)

            joblib.dump(pca_grad, os.path.join(cache_dir, 'pca_model_gradient.joblib'))
            tree_grad = BallTree(features_pca_grad, metric='euclidean')
            joblib.dump(tree_grad, os.path.join(cache_dir, 'search_index_gradient.joblib'))

            meta['pca_explained_variance_gradient'] = float(
                np.sum(pca_grad.explained_variance_ratio_))
    except Exception as e:
        print(f"WARNING: PCA/BallTree build failed ({e}), brute-force only.")

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
                           use_pca=False,
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
    :param use_pca: Use PCA+BallTree for fast approximate search (falls back to
        brute-force if PCA artifacts are missing).
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

        # Compute query feature vector for BRUTE-FORCE (with range masking)
        v_query_bf = extractor.extract(x_query, y_query, custom_range=effective_range, use_gradient=use_gradient)

        # Compute query feature vector for PCA (NO masking - full spectrum)
        v_query_pca = extractor.extract(x_query, y_query, custom_range=None, use_gradient=use_gradient)

        # Load cached spectra
        spectra = np.load(os.path.join(cache_dir, 'spectra.npy'), mmap_mode='r')

        # Resample cube spectra to match query grid if grids differ
        x_query_masked = x_query[mask_query]
        x_cube_masked = x_cube[mask_cube]

        if not np.array_equal(x_query_masked, x_cube_masked):
            # Resample all pixels at once (vectorized along spectral axis)
            cube_masked = spectra[:, :, mask_cube]
            nrows, ncols, n_src = cube_masked.shape
            n_target = mask_query.sum()
            flat = cube_masked.reshape(-1, n_src)
            flat_resampled = resample(flat, n_target, axis=1)
            cube_spectra = flat_resampled.reshape(nrows, ncols, n_target).astype(np.float32)
            x_for_extract = x_query_masked
        else:
            cube_spectra = spectra[:, :, mask_cube]
            x_for_extract = x_cube_masked

        # Extract features from cube spectra
        # For brute-force: use range-masked spectra (already masked above)
        v_cube_bf = extractor.extract(x_for_extract, cube_spectra, custom_range=None, use_gradient=use_gradient)

        # For PCA: use FULL spectrum (no range masking)
        v_cube_pca = extractor.extract(x_cube, spectra, custom_range=None, use_gradient=use_gradient)

        # --- PCA fast path ---
        if use_pca:
            try:
                # Load precomputed PCA model and BallTree index based on gradient mode
                model_suffix = '_gradient' if use_gradient else ''
                pca_model_path = os.path.join(cache_dir, f'pca_model{model_suffix}.joblib')
                search_index_path = os.path.join(cache_dir, f'search_index{model_suffix}.joblib')

                if not os.path.isfile(pca_model_path) or not os.path.isfile(search_index_path):
                    raise FileNotFoundError(f"PCA model or search index not found (use_gradient={use_gradient})")

                pca = joblib.load(pca_model_path)
                tree = joblib.load(search_index_path)

                # Extract features from cube spectra and project to PCA space
                # v_cube_pca is full spectrum (matches training)
                nrows, ncols = v_cube_pca.shape[0], v_cube_pca.shape[1]
                feature_dim = v_cube_pca.shape[2]
                flat_features = v_cube_pca.reshape(nrows * ncols, feature_dim)

                # Project query and cube features to PCA space (using full spectrum features)
                query_pca = pca.transform(v_query_pca.reshape(1, -1))
                cube_pca = pca.transform(flat_features)

                # Perform k-NN search in PCA space
                dists, indices = tree.query(query_pca, k=num_hits)

                # Collect hits with proper coordinate mapping
                sr = meta['sample_rate']
                for dist, flat_idx in zip(dists[0], indices[0]):
                    sy, sx = divmod(int(flat_idx), ncols)
                    all_hits.append({
                        'error': float(dist),
                        'cube_file': cube_filepath,
                        'cube_name': os.path.splitext(
                            os.path.basename(cube_filepath))[0],
                        'x': int(sx * sr),
                        'y': int(sy * sr),
                        'sampled_x': int(sx),
                        'sampled_y': int(sy),
                        'spectrum_y': np.array(spectra[sy, sx, :]),
                        'spectrum_x': np.array(meta['bands']),
                        'cache_dir': cache_dir,
                        'nrows': meta['nrows'],
                        'ncols': meta['ncols'],
                    })
                continue  # skip brute-force for this cube
            except Exception as e:
                print(f"WARNING: PCA search failed ({e}), falling back to brute-force.")

        # --- Brute-force path ---

        # Compute distances using range-masked features: v_cube_bf is (rows, cols, vec_dim), v_query_bf is (vec_dim,)
        errs = v_cube_bf - v_query_bf
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
