import os
import hashlib
import numpy as np


class VectorStore:
    """
    Two-level (memory + disk) content-addressed cache for numpy arrays.

    Keys are SHA-256 hex digests derived from the exact inputs that
    determine a vector. Values are numpy arrays.

    Disk format: individual .npy files in <cache_dir>/vectors/.
    Memory: dict mapping key -> np.ndarray.

    PIPELINE_VERSION must be bumped whenever spectrum_to_vector logic
    changes, to invalidate all cached vectors.
    """

    PIPELINE_VERSION = "1"

    def __init__(self, cache_dir=None):
        """
        :param cache_dir: Directory for .npy files. None = memory-only mode.
        """
        self._memory = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(os.path.join(cache_dir, 'vectors'), exist_ok=True)

    @staticmethod
    def _compute_hash(*parts):
        """Compute SHA-256 from heterogeneous inputs."""
        h = hashlib.sha256()
        h.update(VectorStore.PIPELINE_VERSION.encode())
        for part in parts:
            if isinstance(part, np.ndarray):
                h.update(part.dtype.str.encode())
                h.update(np.array(part.shape).tobytes())
                h.update(part.tobytes())
            elif part is None:
                h.update(b'\x00')
            elif isinstance(part, (tuple, list)):
                for elem in part:
                    h.update(str(elem).encode())
                    h.update(b'\x1f')
            elif isinstance(part, bool):
                h.update(b'\x01' if part else b'\x02')
            else:
                h.update(str(part).encode())
            h.update(b'\x1e')
        return h.hexdigest()

    def get(self, key):
        """Retrieve cached vector. Returns None on miss."""
        if key in self._memory:
            return self._memory[key]
        if self._cache_dir:
            path = os.path.join(self._cache_dir, 'vectors', f'{key}.npy')
            if os.path.exists(path):
                arr = np.load(path)
                self._memory[key] = arr
                return arr
        return None

    def put(self, key, value):
        """Store vector in memory and optionally on disk."""
        self._memory[key] = value
        if self._cache_dir:
            path = os.path.join(self._cache_dir, 'vectors', f'{key}.npy')
            np.save(path, value)

    def clear_memory(self):
        """Drop in-memory cache. Disk cache remains."""
        self._memory.clear()

    def clear_all(self):
        """Drop both in-memory and disk caches."""
        self._memory.clear()
        if self._cache_dir:
            vdir = os.path.join(self._cache_dir, 'vectors')
            if os.path.isdir(vdir):
                for f in os.listdir(vdir):
                    if f.endswith('.npy'):
                        os.remove(os.path.join(vdir, f))

    def make_db_vector_key(self, x_query, x_db, y_db, custom_range, use_gradient):
        """
        Cache key for a DB spectrum's feature vector.

        Excludes y_query — v2's computation path (overlap -> mask ->
        resample -> extract) does not depend on the query's intensity values.
        """
        return self._compute_hash(x_query, x_db, y_db, custom_range, use_gradient)

    def make_query_vector_key(self, x_query, y_query, effective_range, use_gradient):
        """Cache key for the query spectrum's feature vector."""
        return self._compute_hash(x_query, y_query, effective_range, use_gradient)
