import abc

from hyperlyse.feature_extractor import FeatureExtractor


class VectorProvider(abc.ABC):
    """
    Abstract interface for providing pre-processed feature vectors
    from a collection of spectra.

    Phase 1: structural placeholder -- not yet wired into search_spectrum.
    Phase 2+: will be used by search_spectrum to iterate over vector sources
    (JDX database, pre-computed cube vectors, etc.).
    """

    @abc.abstractmethod
    def get_vectors(self, config):
        """
        Yield or return feature vectors for all spectra managed by this provider.

        :param config: dict with keys:
            - 'custom_range': (x_min, x_max) or None
            - 'use_gradient': bool
        :return: list of dicts, each with:
            - 'vector': np.ndarray (the feature vector)
            - 'spectrum': Spectrum object (original data + metadata)
        """
        raise NotImplementedError


class JDXVectorProvider(VectorProvider):
    """
    Provides feature vectors from a list of JDX-loaded Spectrum objects.

    :param spectra: list of Spectrum objects (e.g. database.spectra)
    """

    def __init__(self, spectra):
        self._spectra = spectra
        self._extractor = FeatureExtractor()

    def get_vectors(self, config):
        """
        Compute and return feature vectors for each spectrum.

        :param config: dict with keys 'custom_range' and 'use_gradient'
        :return: list of dicts with 'vector' and 'spectrum' keys
        """
        custom_range = config.get('custom_range', None)
        use_gradient = config.get('use_gradient', False)

        results = []
        for spectrum in self._spectra:
            vector = self._extractor.extract(
                spectrum.x,
                spectrum.y,
                custom_range=custom_range,
                use_gradient=use_gradient,
            )
            results.append({
                'vector': vector,
                'spectrum': spectrum,
            })
        return results
