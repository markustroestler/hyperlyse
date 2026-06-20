import os
import gc
import stat
import json
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import joblib
from scipy.signal import resample
from sklearn.decomposition import PCA

from hyperlyse.database import spectrum_to_vector


# Bump whenever the on-disk cache format changes so stale caches auto-rebuild.
# "3": replaced the BallTree search_index.joblib with pca_features.npy.
PIPELINE_VERSION = "3"
CUBE_DATA_EXTENSIONS = {'.raw', '.dat', '.bil'}
REFERENCE_PREFIXES = ('DARKREF_', 'WHITEREF_')
REFLECTANCE_PREFIX = 'REFLECTANCE_'
# Specim IQ stores the raw cube and the calibrated reflectance product in
# sibling subfolders of a capture; collapse them so a scene loads only once.
SCENE_SUBFOLDERS = ('capture', 'results')

# Cubes are independent of each other, so analysis and search both fan out over
# them. We use threads (not processes) because the heavy steps — spectral file
# I/O, np.load/np.save, the LAPACK-backed PCA fit, and the BallTree build — all
# release the GIL, and threads keep the cube_class dependency injection used by
# the tests working without any pickling constraints.
_CPU_COUNT = os.cpu_count() or 1
DEFAULT_ANALYSIS_WORKERS = max(1, min(4, _CPU_COUNT))
DEFAULT_SEARCH_WORKERS = max(1, min(8, _CPU_COUNT))

# Fitting PCA on every pixel of a large cube is wasteful — the principal
# components are statistically stable from a random subsample. We still
# transform every pixel (the BallTree must contain them all), only the fit is
# subsampled. Deterministic seed so caches are reproducible.
PCA_FIT_SAMPLE = 20000
_PCA_FIT_SEED = 0

# Loaded PCA models / BallTree indices are cached in memory across searches,
# keyed by (path, mtime), so repeated interactive searches don't re-read every
# cube's joblib artifacts from disk. Bounded to avoid unbounded growth.
_model_cache = {}
_model_cache_lock = threading.Lock()
_MODEL_CACHE_MAX = 512


def _load_joblib_cached(path):
    """Load a joblib artifact, caching by (path, mtime). Returns None if absent."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime)
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    obj = joblib.load(path)
    with _model_cache_lock:
        if len(_model_cache) >= _MODEL_CACHE_MAX:
            _model_cache.clear()
        _model_cache[key] = obj
    return obj


def _load_npy_cached(path):
    """Load a .npy array into RAM, caching by (path, mtime). Returns None if absent."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime)
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    arr = np.load(path)
    with _model_cache_lock:
        if len(_model_cache) >= _MODEL_CACHE_MAX:
            _model_cache.clear()
        _model_cache[key] = arr
    return arr


def discover_cubes(folder, include_subfolders=False):
    """
    Scan a folder for hyperspectral cube data files.

    Returns a sorted list of absolute file paths. Excludes .hdr files,
    dark/white reference files, and files in .hyperlyse_cache.

    When both a calibrated reflectance product (REFLECTANCE_<id>) and the raw
    capture (<id>) exist for the same scene, only the reflectance file is
    returned — it is already calibrated by Specim IQ Studio, so the raw is
    redundant. The raw is used only when no reflectance product exists.
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

    results = _prefer_reflectance(results)
    results.sort()
    return results


def _is_reflectance_file(filename):
    """Check if a filename is a calibrated reflectance product."""
    return os.path.basename(filename).startswith(REFLECTANCE_PREFIX)


def _scene_key(filepath):
    """
    Compute a grouping key that pairs the raw capture and its reflectance
    product. The capture id is the basename without extension and without a
    leading REFLECTANCE_ prefix; the scene root is the parent of a
    capture/results subfolder when present, otherwise the file's own folder.
    """
    directory = os.path.dirname(filepath)
    capture_id, _ = os.path.splitext(os.path.basename(filepath))
    if capture_id.startswith(REFLECTANCE_PREFIX):
        capture_id = capture_id[len(REFLECTANCE_PREFIX):]
    if os.path.basename(directory).lower() in SCENE_SUBFOLDERS:
        scene_root = os.path.dirname(directory)
    else:
        scene_root = directory
    return (scene_root, capture_id)


def _prefer_reflectance(filepaths):
    """
    Collapse raw/reflectance variants of the same scene to a single file,
    preferring the reflectance product when one exists.
    """
    scenes = {}
    for path in filepaths:
        scenes.setdefault(_scene_key(path), []).append(path)

    chosen = []
    for variants in scenes.values():
        reflectance = [p for p in variants if _is_reflectance_file(p)]
        chosen.extend(reflectance if reflectance else variants)
    return chosen


def _scene_key_normalized(filepath):
    """
    Scene key with a normalized (absolute, case-folded) scene root, suitable for
    comparing two paths that may use different separators or refer to the raw
    capture vs the reflectance product of the same scene. Returns None for an
    empty path.
    """
    if not filepath:
        return None
    scene_root, capture_id = _scene_key(filepath)
    return (os.path.normcase(os.path.abspath(scene_root)), capture_id)


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


def _is_cache_valid(cache_dir, cube_filepath, sample_rate, check_sample_rate=True):
    """Check if a cube's cache is valid and complete.

    :param check_sample_rate: When True (analysis phase) the cache is only valid
        if it was built at the requested sample_rate, so changing the rate forces
        a re-analysis. The cross-cube search passes False: it reads each cube's
        actual sample_rate from metadata when mapping hits, so a cache built at
        any rate is usable and changing the UI rate must not blank out results.
    """
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
    if check_sample_rate and meta.get('sample_rate') != sample_rate:
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


def _fit_transform_pca(flat_features, n_components):
    """Fit a PCA and project all rows of flat_features.

    For large pixel counts the fit is done on a deterministic random subsample
    (the components barely change), but every row is transformed so the BallTree
    holds all pixels. Returns (fitted_pca, projected_features).
    """
    pca = PCA(n_components=n_components, svd_solver='auto')
    n_rows = flat_features.shape[0]
    if n_rows > PCA_FIT_SAMPLE:
        rng = np.random.default_rng(_PCA_FIT_SEED)
        fit_idx = rng.choice(n_rows, PCA_FIT_SAMPLE, replace=False)
        pca.fit(flat_features[fit_idx])
        projected = pca.transform(flat_features)
    else:
        projected = pca.fit_transform(flat_features)
    return pca, projected


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
    _robust_rmtree(cache_dir)
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

            # Train PCA on RAW features (use_gradient=False). Raw features are the
            # spectra themselves, so reshape the cube directly instead of letting
            # spectrum_to_vector copy it. We persist the PCA model (to project the
            # query at search time) and the projected pixel features. A single
            # query needs only a vectorized k-NN over these features, so we skip
            # the (expensive to build) BallTree entirely.
            flat_features_raw = spectra.reshape(n_pixels, -1)
            pca_raw, features_pca_raw = _fit_transform_pca(flat_features_raw, n_components)

            joblib.dump(pca_raw, os.path.join(cache_dir, 'pca_model.joblib'))
            np.save(os.path.join(cache_dir, 'pca_features.npy'),
                    features_pca_raw.astype(np.float32))

            meta['pca_components'] = n_components
            meta['pca_explained_variance_raw'] = float(
                np.sum(pca_raw.explained_variance_ratio_))

            # Train PCA on GRADIENT features (use_gradient=True)
            features_grad = spectrum_to_vector(x_bands, spectra, custom_range=None, use_gradient=True)
            flat_features_grad = features_grad.reshape(n_pixels, -1)
            pca_grad, features_pca_grad = _fit_transform_pca(flat_features_grad, n_components)

            joblib.dump(pca_grad, os.path.join(cache_dir, 'pca_model_gradient.joblib'))
            np.save(os.path.join(cache_dir, 'pca_features_gradient.npy'),
                    features_pca_grad.astype(np.float32))

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
                  cube_class=None, progress_callback=None,
                  discovered_callback=None, max_workers=None):
    """
    Discover and analyze all cubes in a folder.

    Cubes are analyzed concurrently (each is independent and writes to its own
    cache directory). Progress callbacks fire once per cube as it finishes,
    carrying a monotonically increasing completion index; the returned results
    preserve discovery order.

    :param cube_folder: Root folder containing cubes.
    :param sample_rate: Spatial sampling rate.
    :param include_subfolders: Whether to recurse into subdirectories.
    :param cube_class: The Cube class (for testing).
    :param progress_callback: Optional callable(current_index, total, cube_name, elapsed_per_cube, skipped).
    :param discovered_callback: Optional callable(total) invoked right after
        discovery, before any analysis begins.
    :param max_workers: Thread pool size. Defaults to DEFAULT_ANALYSIS_WORKERS,
        capped at the number of cubes. Pass 1 to force sequential analysis.
    :return: List of (cube_filepath, cache_dir) tuples for analyzed/cached cubes.
    """
    cube_files = discover_cubes(cube_folder, include_subfolders)
    total = len(cube_files)
    if discovered_callback:
        discovered_callback(total)
    if total == 0:
        return []

    if max_workers is None:
        max_workers = DEFAULT_ANALYSIS_WORKERS
    max_workers = max(1, min(max_workers, total))

    results = [None] * total
    lock = threading.Lock()
    progress_state = {'done': 0, 'elapsed_sum': 0.0}

    def process_one(i, cube_filepath):
        cube_name = os.path.basename(cube_filepath)
        cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)

        if _is_cache_valid(cache_dir, cube_filepath, sample_rate):
            results[i] = (cube_filepath, cache_dir)
            elapsed, skipped = 0.0, True
        else:
            t0 = time.time()
            skipped = False
            try:
                cache_dir = analyze_cube(cube_filepath, cube_folder, sample_rate, cube_class)
                results[i] = (cube_filepath, cache_dir)
            except Exception as e:
                print(f"Error analyzing {cube_filepath}: {e}")
            elapsed = time.time() - t0

        if progress_callback:
            with lock:
                done = progress_state['done']
                progress_state['done'] += 1
                if not skipped:
                    progress_state['elapsed_sum'] += elapsed
                avg_time = (progress_state['elapsed_sum'] / progress_state['done']
                            if progress_state['done'] else 0)
                progress_callback(done, total, cube_name, avg_time, skipped=skipped)

    if max_workers == 1:
        for i, cube_filepath in enumerate(cube_files):
            process_one(i, cube_filepath)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, i, cf)
                       for i, cf in enumerate(cube_files)]
            for f in futures:
                f.result()

    return [r for r in results if r is not None]


def _overlap_masks(x_query, x_cube, custom_range):
    """Compute the overlapping wavelength range and the query/cube band masks."""
    lambda_min = max(x_query[0], x_cube[0])
    lambda_max = min(x_query[-1], x_cube[-1])
    if custom_range is not None:
        lambda_min = max(lambda_min, custom_range[0])
        lambda_max = min(lambda_max, custom_range[1])
    mask_query = np.logical_and(x_query >= lambda_min, x_query <= lambda_max)
    mask_cube = np.logical_and(x_cube >= lambda_min, x_cube <= lambda_max)
    return (lambda_min, lambda_max), mask_query, mask_cube


def _score_block(x_query, y_query, x_cube, block, mask_query, mask_cube,
                 effective_range, use_gradient, squared_errs):
    """Mean per-pixel error between the query and a block of cube spectra.

    ``block`` is a (rows, cols, nbands_full) array of full-spectrum cube spectra;
    the return is a (rows, cols) error map. This is the single source of truth for
    the comparison metric — both the brute-force full-cube scan and the PCA
    candidate re-ranking call it, so their errors are on the same scale.
    """
    v_query = spectrum_to_vector(x_query, y_query,
                                 custom_range=effective_range, use_gradient=use_gradient)

    x_query_masked = x_query[mask_query]
    x_cube_masked = x_cube[mask_cube]
    block_masked = block[:, :, mask_cube]

    if not np.array_equal(x_query_masked, x_cube_masked):
        nrows, ncols, n_src = block_masked.shape
        n_target = int(mask_query.sum())
        flat = block_masked.reshape(-1, n_src)
        flat_resampled = resample(flat, n_target, axis=1)
        block_for_extract = flat_resampled.reshape(nrows, ncols, n_target).astype(np.float32)
        x_for_extract = x_query_masked
    else:
        block_for_extract = block_masked
        x_for_extract = x_cube_masked

    v_cube = spectrum_to_vector(x_for_extract, block_for_extract,
                                custom_range=None, use_gradient=use_gradient)
    errs = v_cube - v_query
    errs = np.power(errs, 2) if squared_errs else np.abs(errs)
    return np.mean(errs, axis=2)


def _top_k_flat(error_map, k):
    """Indices of the k smallest values in error_map, ascending. O(n) selection."""
    flat = error_map.reshape(-1)
    k = min(k, flat.shape[0])
    part = np.argpartition(flat, k - 1)[:k]
    return part[np.argsort(flat[part])]


def _make_hit(error_val, cube_filepath, sy, sx, sr, spectra, meta, cache_dir):
    """Build a hit dict for a sampled pixel (sy, sx). Copies the spectrum out of
    the mmap so no file handle is retained (Windows can't delete an open mmap)."""
    return {
        'error': float(error_val),
        'cube_file': cube_filepath,
        'cube_name': os.path.splitext(os.path.basename(cube_filepath))[0],
        'x': int(sx * sr),
        'y': int(sy * sr),
        'sampled_x': int(sx),
        'sampled_y': int(sy),
        'spectrum_y': np.array(spectra[sy, sx, :]),
        'spectrum_x': np.array(meta['bands']),
        'cache_dir': cache_dir,
        'nrows': meta['nrows'],
        'ncols': meta['ncols'],
    }


def _search_one_cube(cube_filepath, cache_dir, meta, x_query, y_query,
                     custom_range, use_gradient, squared_errs, num_hits, use_pca):
    """Search a single cached cube and return its list of hit dicts (may be empty)."""
    x_cube = np.array(meta['bands'])
    effective_range, mask_query, mask_cube = _overlap_masks(x_query, x_cube, custom_range)
    if mask_query.sum() < 2 or mask_cube.sum() < 2:
        return []

    spectra = np.load(os.path.join(cache_dir, 'spectra.npy'), mmap_mode='r')
    sr = meta['sample_rate']
    hits = []

    # --- PCA fast path: prefilter candidates with the prebuilt BallTree, then
    # re-rank those few candidates with the exact metric so the reported error is
    # comparable to the brute-force path. ---
    if use_pca:
        try:
            model_suffix = '_gradient' if use_gradient else ''
            pca_model_path = os.path.join(cache_dir, f'pca_model{model_suffix}.joblib')
            pca_features_path = os.path.join(cache_dir, f'pca_features{model_suffix}.npy')
            if not os.path.isfile(pca_model_path) or not os.path.isfile(pca_features_path):
                raise FileNotFoundError(
                    f"PCA model or features not found (use_gradient={use_gradient})")

            pca = _load_joblib_cached(pca_model_path)
            features_pca = _load_npy_cached(pca_features_path)  # (n_pixels, n_components)

            ncols = spectra.shape[1]
            n_pixels = spectra.shape[0] * ncols

            # Query projected into the full-spectrum PCA space the features live in.
            v_query_full = spectrum_to_vector(x_query, y_query,
                                              custom_range=None, use_gradient=use_gradient)
            query_pca = pca.transform(v_query_full.reshape(1, -1))

            # Vectorized k-NN prefilter in PCA space (squared distance is enough to
            # rank). Over-fetch candidates; this distance is an approximation, the
            # exact re-rank below decides the final ordering.
            diff = features_pca - query_pca  # (n_pixels, n_components)
            pca_dist = np.einsum('ij,ij->i', diff, diff)
            k_fetch = min(n_pixels, max(num_hits * 4, num_hits + 10))
            cand = np.argpartition(pca_dist, k_fetch - 1)[:k_fetch]

            cand_sy, cand_sx = np.divmod(cand.astype(int), ncols)
            cand_block = np.array(spectra[cand_sy, cand_sx, :])[:, None, :]  # (k, 1, nbands)
            cand_err = _score_block(x_query, y_query, x_cube, cand_block,
                                    mask_query, mask_cube, effective_range,
                                    use_gradient, squared_errs)[:, 0]

            order = np.argsort(cand_err)[:num_hits]
            for j in order:
                hits.append(_make_hit(cand_err[j], cube_filepath,
                                      int(cand_sy[j]), int(cand_sx[j]), sr,
                                      spectra, meta, cache_dir))
            return hits
        except Exception as e:
            print(f"WARNING: PCA search failed ({e}), falling back to brute-force.")
            hits = []

    # --- Brute-force path: exact metric over every pixel. ---
    error_map = _score_block(x_query, y_query, x_cube, spectra,
                             mask_query, mask_cube, effective_range,
                             use_gradient, squared_errs)
    for idx in _top_k_flat(error_map, num_hits):
        sy, sx = np.unravel_index(idx, error_map.shape)
        hits.append(_make_hit(error_map[sy, sx], cube_filepath,
                              int(sy), int(sx), sr, spectra, meta, cache_dir))
    return hits


def search_in_cached_cubes(cube_folder, x_query, y_query,
                           sample_rate=1, include_subfolders=False,
                           custom_range=None, use_gradient=False,
                           squared_errs=True, num_hits=3,
                           use_pca=False, exclude_cube_file=None,
                           progress_callback=None, max_workers=None):
    """
    Search across all cached cubes for spectra similar to the query.

    Cubes are searched concurrently. Progress callbacks fire once per cube as it
    finishes, carrying a monotonically increasing completion index.

    :param cube_folder: Root folder with cached cubes.
    :param x_query: Wavelength array of the query spectrum.
    :param y_query: Intensity array (1D only).
    :param sample_rate: Must match the sample_rate used during analysis.
    :param include_subfolders: Whether subfolders were included in analysis.
    :param custom_range: (x_min, x_max) wavelength range for comparison. Honored
        in both brute-force and PCA modes (PCA prefilters on the full spectrum,
        then re-ranks candidates within this range).
    :param use_gradient: Compare gradients instead of raw spectra.
    :param squared_errs: Use squared errors.
    :param num_hits: Number of top hits to return.
    :param use_pca: Use the prebuilt PCA+BallTree index to prefilter candidates
        (falls back to brute-force if PCA artifacts are missing). Results are
        re-ranked with the exact metric, so errors are comparable to brute-force.
    :param exclude_cube_file: Optional cube file path to exclude from the search
        (e.g. the cube currently open in the UI). Matched by scene, so the raw
        capture and its reflectance product are treated as the same cube.
    :param progress_callback: Optional callable(current, total, cube_name, avg_time).
    :param max_workers: Thread pool size. Defaults to DEFAULT_SEARCH_WORKERS,
        capped at the number of cubes. Pass 1 to force sequential search.
    :return: List of hit dicts sorted by error, up to num_hits entries.
    """
    x_query = np.array(x_query)
    y_query = np.array(y_query)

    if len(y_query.shape) != 1:
        raise ValueError("y_query must be 1D for cube search")

    cube_files = discover_cubes(cube_folder, include_subfolders)
    exclude_key = _scene_key_normalized(exclude_cube_file)
    total = len(cube_files)
    if total == 0:
        return []

    if max_workers is None:
        max_workers = DEFAULT_SEARCH_WORKERS
    max_workers = max(1, min(max_workers, total))

    all_hits = []
    hits_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_state = {'done': 0, 't_start': time.time()}

    def process_one(cube_filepath):
        cube_name = os.path.splitext(os.path.basename(cube_filepath))[0]
        hits = []
        try:
            # Skip the excluded scene. Comparison is by scene key (not exact path)
            # so opening the raw capture still excludes its reflectance product.
            excluded = (exclude_key is not None
                        and _scene_key_normalized(cube_filepath) == exclude_key)
            cache_dir = _cache_dir_for_cube(cube_folder, cube_filepath)

            # Search uses whatever cache exists, regardless of the current UI
            # sample rate; hit coordinates are mapped using each cube's cached rate.
            if not excluded and _is_cache_valid(cache_dir, cube_filepath,
                                                sample_rate, check_sample_rate=False):
                with open(os.path.join(cache_dir, 'metadata.json'), 'r') as f:
                    meta = json.load(f)
                hits = _search_one_cube(cube_filepath, cache_dir, meta,
                                        x_query, y_query, custom_range,
                                        use_gradient, squared_errs, num_hits, use_pca)
        finally:
            if hits:
                with hits_lock:
                    all_hits.extend(hits)
            with progress_lock:
                done = progress_state['done']
                progress_state['done'] += 1
                if progress_callback is not None:
                    elapsed = time.time() - progress_state['t_start']
                    avg_time = elapsed / (done + 1)
                    progress_callback(done, total, cube_name, avg_time)

    if max_workers == 1:
        for cube_filepath in cube_files:
            process_one(cube_filepath)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, cf) for cf in cube_files]
            for f in futures:
                f.result()

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


def _robust_rmtree(path, retries=5, delay=0.2):
    """
    Delete a directory tree, tolerating transient Windows file locks.

    numpy memory-maps (np.load(..., mmap_mode='r')) keep an OS handle on the
    backing .npy file until the array is garbage-collected; on Windows an open
    handle blocks deletion with WinError 32. We force a collection first, then
    retry rmtree, clearing read-only flags on any file that resists.
    """
    if not os.path.isdir(path):
        return

    # Release any lingering memmap handles before we start deleting.
    gc.collect()

    def on_error(func, target, exc_info):
        # Clear a possible read-only attribute and retry once inline.
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            raise

    last_exc = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=on_error)
            return
        except OSError as e:
            last_exc = e
            gc.collect()
            time.sleep(delay)

    # Final attempt; let the exception propagate if it still fails.
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=on_error)
    if last_exc is not None and os.path.isdir(path):
        raise last_exc


def reset_cache(cube_folder):
    """Delete all cached cube analysis data."""
    cache_dir = os.path.join(cube_folder, '.hyperlyse_cache', 'cube_vectors')
    _robust_rmtree(cache_dir)


def load_rgb_preview(cache_dir):
    """Load the RGB preview image from a cube's cache directory."""
    path = os.path.join(cache_dir, 'rgb_preview.npy')
    if os.path.isfile(path):
        return np.load(path)
    return None


def load_metadata(cache_dir):
    """Load the metadata dict from a cube's cache directory, or None."""
    path = os.path.join(cache_dir, 'metadata.json')
    if os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def load_spectra(cache_dir):
    """Load the cached (spatially sampled) spectra cube into memory.

    Reads fully into RAM rather than memory-mapping so the caller can hold the
    array without keeping an OS file handle open (on Windows an open mmap blocks
    deletion of the backing .npy during a cache reset). Returns None if missing.
    """
    path = os.path.join(cache_dir, 'spectra.npy')
    if os.path.isfile(path):
        return np.load(path)
    return None
