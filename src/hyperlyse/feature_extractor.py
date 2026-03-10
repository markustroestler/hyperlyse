import numpy as np


# NOTE (caching context):
# This function is pure computation — no caching awareness.
# Caching is handled one layer up, in search_spectrum (database.py),
# via VectorStore. The cache keys use the ORIGINAL (pre-resampling)
# DB spectrum data, so v2 vectors are reused across pixel selections.
# compare_spectra (static) remains uncached — single-call paths
# (error map, single comparison) don't benefit from caching.
def spectrum_to_vector(x, y, custom_range=None, use_gradient=False):
    """
    Prepares a spectrum (or spectral cube) for comparison by applying
    wavelength range masking and optional gradient computation.

    :param x: 1D wavelength array
    :param y: intensity array - 1D (bands,) or 3D cube (rows, cols, bands)
    :param custom_range: (x_min, x_max) wavelength range to keep
    :param use_gradient: if True, return gradient instead of raw values
    :return: processed spectrum/cube with masking and optional gradient applied
    """
    print('spectrum_to_vector: applying custom range and gradient (if selected)')

    x = np.array(x)
    y = np.array(y)
    is_cube = len(y.shape) == 3

    if custom_range is not None:
        mask = np.logical_and(x >= custom_range[0], x <= custom_range[1])
        if is_cube:
            y = y[:, :, mask]
        else:
            y = y[mask]

    if use_gradient:
        if is_cube:
            y = np.gradient(y, axis=2)
        else:
            y = np.gradient(y)

    return y


class FeatureExtractor:
    """
    Extracts comparison-ready feature vectors from spectra.

    Stateless thin wrapper around spectrum_to_vector().
    No caching here — caching lives in VectorStore, orchestrated
    by search_spectrum in database.py.
    """

    def extract(self, x, y, custom_range=None, use_gradient=False):
        """
        Delegates to spectrum_to_vector(). Same parameters, same return value.
        """
        return spectrum_to_vector(x, y, custom_range=custom_range, use_gradient=use_gradient)
