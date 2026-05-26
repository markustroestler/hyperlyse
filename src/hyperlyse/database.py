import os
import re
import json
import numpy as np
import matplotlib.image
from scipy.signal import resample
import collections


class Metadata:
    def __init__(self,
                 id,
                 description='',
                 source_object='',
                 source_file='',
                 source_coordinates='',
                 device_info='',
                 intensity=''
                 ):
        self.id = id
        self.description = description
        self.source_object = source_object
        self.source_file = source_file
        self.source_coordinates = source_coordinates
        self.device_info = device_info
        self.intensity = intensity


class Spectrum:
    def __init__(self, x, y, metadata: Metadata):
        self.x = np.array(x)
        self.y = np.array(y)
        self.metadata = metadata

    def display_string(self, with_description=False, separator=' | '):
        values = [self.metadata.source_object,
                  self.metadata.id]
        if with_description:
            values.append(self.metadata.description)
        values = [v for v in values if v]
        return separator.join(values)

    def save_dpt(self, file_name):
        data = np.transpose([np.float32(self.x), np.float32(self.y)])
        np.savetxt(file_name, data, fmt='%.4f', delimiter=',')

    def save_jcamp(self, file_name):
        # prepare output file content
        data = collections.OrderedDict()  # in jcamp, order of elements is kind of important..
        data['##TITLE'] = f'{self.metadata.id} | {self.metadata.source_object}'
        data['##JCAMP-DX'] = "5.1"
        data['##DATA TYPE'] = "UV/VIS SPECTRUM"
        data['##ORIGIN'] = "CIMA"
        data['##OWNER'] = "CIMA"

        data['##DATA CLASS'] = 'XYDATA'
        data['##SPECTROMETER/DATASYSTEM'] = self.metadata.device_info
        data['##SOURCE REFERENCE'] = f'{self.metadata.source_file} | {self.metadata.source_coordinates}'
        data['##SAMPLE DESCRIPTION'] = f'{self.metadata.description} | {self.metadata.intensity}'
        ##INSTRUMENTAL PARAMETERS=(STRING).This optional field is a list of pertinent instrumental settings. Only
        # settings which are essential for applications should be included.
        data['##SAMPLING PROCEDURE'] = "MODE=reflection"
        # First entry in this field should be MODE of observation (transmission,
        # specular reflection, PAS, matrix isolation, photothermal beam deflection, etc.), followed by appropriate
        # additional information, i.e., name and model of accessories, cell thickness, and window material for
        # fixed liquid cells, ATR plate material, angle and cone of incidence, and effective number of reflections
        # for ATR measurements, polarization, and special modulation techniques, as discussed by Grasselli et al.
        # data['##DATA PROCESSING'] = ""
        # (TEXT). Description of background correction, smoothing, subtraction,
        # deconvolution procedures, apodization function, zero - fill, or other data processing, together
        # with reference to original spectra used for subtractions.

        vx = np.float32(self.x)
        vy = np.float32(self.y)

        data['##DELTAX'] = (vx[-1] - vx[0]) / (len(vx) - 1)
        data['##XUNITS'] = "NANOMETERS"
        data['##YUNITS'] = "REFLECTANCE"
        data['##XFACTOR'] = 1.0
        data['##YFACTOR'] = 1.0

        data['##FIRSTX'] = vx[0]
        data['##LASTX'] = vx[-1]
        data['##NPOINTS'] = len(vx)
        data['##FIRSTY'] = vy[0]
        data['##XYDATA'] = [xy for xy in zip(vx, vy)]

        data['##END'] = ''

        # write the file
        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            for k, v in data.items():
                if k == "##XYDATA":
                    f.write('##XYDATA= (X++(Y..Y))\n')
                    for x, y in v:
                        f.write('%s %s\n' % (str(x), str(y)))
                else:
                    f.write('%s= %s\n' % (k.replace('_', ' '), str(v)))

    @staticmethod
    def __jcamp_line_to_key_value(line):
        result = re.search(r'##(.*)= (.*)', line)
        if result:
            return (result.group(1), result.group(2))
        else:
            return None

    @staticmethod
    def __jcamp_split_multi_values(combined_values: str, expected_n_values=0, separator='|'):
        values = [v.strip(' ') for v in combined_values.split(separator)]
        if expected_n_values > 0:
            while len(values) < expected_n_values:
                values.append('')
        return values

    @staticmethod
    def load_jcamp(file):
        """
        Only works with files produced by __save_jcamp
        :param file:
        :return:
        """
        with open(file, 'r') as f:
            lines = f.read().splitlines()

        start_xy_data = False
        metadata = Metadata('')
        x = []
        y = []
        for line in lines:
            kv = Spectrum.__jcamp_line_to_key_value(line)
            if kv is None:
                if start_xy_data:
                    vx, vy = Spectrum.__jcamp_split_multi_values(line, 2, ' ')
                    try:
                        x.append(float(vx))
                        y.append(float(vy))
                    except:
                        pass
            else:
                k, v = kv
                if k == 'TITLE':
                    id, src_obj = Spectrum.__jcamp_split_multi_values(v, 2)
                    metadata.id = id
                    metadata.source_object = src_obj
                elif k == 'SPECTROMETER/DATASYSTEM':
                    metadata.device_info = v
                elif k == 'SOURCE REFERENCE':
                    src_file, src_coords = Spectrum.__jcamp_split_multi_values(v, 2)
                    metadata.source_file = src_file
                    metadata.source_coordinates = src_coords
                elif k == 'SAMPLE DESCRIPTION':
                    description, intensity = Spectrum.__jcamp_split_multi_values(v, 2)
                    metadata.description = description
                    metadata.intensity = intensity
                elif k == 'XYDATA':
                    start_xy_data = True
                elif k == 'END':
                    break
        return Spectrum(np.array(x),
                        np.array(y),
                        metadata)



# Backward compatibility: spectrum_to_vector moved to feature_extractor.py
from hyperlyse.feature_extractor import spectrum_to_vector  # noqa: F401
from hyperlyse.feature_extractor import FeatureExtractor
from hyperlyse.vector_store import VectorStore


class Database:

    def __init__(self, root=''):
        self.root = root
        #self.data = None
        #self.file_data = None
        self.spectra = []
        self._extractor = FeatureExtractor()
        cache_dir = os.path.join(root, '.hyperlyse_cache') if root else None
        self._store = VectorStore(cache_dir)
        self.refresh_from_disk()

    def refresh_from_disk(self, new_root=''):
        if new_root:
            self.root = new_root
            cache_dir = os.path.join(new_root, '.hyperlyse_cache')
            self._store = VectorStore(cache_dir)
        if self.root:
            self.spectra = []
            for root, dirs, files in os.walk(self.root):
                for f in files:
                    base, ext = os.path.splitext(f)
                    if ext in ['.dx', '.jdx', '.jcm']:
                        try:
                            spectrum = Spectrum.load_jcamp(os.path.join(root, f))
                            self.spectra.append(spectrum)
                        except:
                            print(f'Error loading {os.path.join(root, f)}')
                            pass

    @staticmethod
    def compare_spectra(x1, y1,
                        x2, y2,
                        custom_range=None,
                        use_gradient=False,
                        squared_errs=True):
        """
        compares 2 spectra
        :param x1: np.array, wavelength array of spectrum 1
        :param y1: np.array, intensity array of spectrum 1 - can be 1d (simple spectrum) or 3d (cube)
        :param x2: np.array, wavelength array of spectrum 2
        :param y2: np.array, intensity array of spectrum 2 - must be 1d, is re-sampled if required
        :param custom_range: (x_min, x_max), a custom range of wavelengths used for comparison
        :param use_gradient: compare gradients instead of absolute differences
        :param squared_errs: use squared differences (or absolute differences)
        :return: mean error/distance; scalar or 2d np.array, depending on shape of y1
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        y1 = np.array(y1)
        y2 = np.array(y2)

        is_cube = len(y1.shape) == 3

        # Compute effective overlapping wavelength range
        lambda_min = max(x1[0], x2[0])
        lambda_max = min(x1[-1], x2[-1])
        if custom_range is not None:
            lambda_min = max(lambda_min, custom_range[0])
            lambda_max = min(lambda_max, custom_range[1])

        # Check for sufficient overlap
        mask1 = np.logical_and(x1 >= lambda_min, x1 <= lambda_max)
        mask2 = np.logical_and(x2 >= lambda_min, x2 <= lambda_max)

        if is_cube:
            y1_masked_size = y1[:, :, mask1].size
        else:
            y1_masked_size = y1[mask1].size
        y2_masked_size = y2[mask2].size

        if y1_masked_size < 2 > y2_masked_size:
            print('WARNING: compared spectra do not have sufficient overlap. Returning None')
            return None

        # Resample y2 to match y1's wavelength grid if they differ
        if not np.array_equal(x1[mask1], x2[mask2]):
            y2 = resample(y2[mask2], mask1.sum())
            x2 = x1[mask1]

        # Prepare comparison vectors
        # INVARIANT: effective_range must match the (lambda_min, lambda_max) used
        # for mask1/mask2 above. spectrum_to_vector will recompute a mask from this
        # range — for v1 it replicates mask1 exactly, and for v2 (after resample)
        # it is an identity no-op since x2 already equals x1[mask1].
        effective_range = (lambda_min, lambda_max)
        _extractor = FeatureExtractor()
        v1 = _extractor.extract(x1, y1, effective_range, use_gradient)
        v2 = _extractor.extract(x2, y2, effective_range, use_gradient)

        errs = v1 - v2

        if squared_errs:
            errs = np.power(errs, 2)
        else:
            errs = np.abs(errs)

        if is_cube:
            return np.mean(errs, axis=2)
        else:
            return np.mean(errs)

    @staticmethod
    def compare_spectra_old(x1, y1,
                        x2, y2,
                        custom_range=None,
                        use_gradient=False,
                        squared_errs=True):
        """
        compares 2 spectra
        :param x1: np.array, wavelength array of spectrum 1
        :param y1: np.array, intensity array of spectrum 1 - can be 1d (simple spectrum) or 3d (cube)
        :param x2: np.array, wavelength array of spectrum 2
        :param y2: np.array, intensity array of spectrum 2 - must be 1d, is re-sampled if required
        :param custom_range: (x_min, x_max), a custom range of wavelengths used for comparison
        :param use_gradient: compare gradients instead of absolute differences
        :param squared_errs: use squared differences (or absolute differences)
        :return: mean error/distance; scalar or 2d np.array, depending on shape of y1
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        y1 = np.array(y1)
        y2 = np.array(y2)

        is_cube = len(y1.shape) == 3

        lambda_min = max(x1[0], x2[0])
        lambda_max = min(x1[-1], x2[-1])
        if custom_range is not None:
            lambda_min = max(lambda_min, custom_range[0])
            lambda_max = min(lambda_max, custom_range[1])

        mask1 = np.logical_and(x1 >= lambda_min, x1 <= lambda_max)
        if is_cube:
            y1_masked = y1[:, :, mask1]
        else:
            y1_masked = y1[mask1]

        mask2 = np.logical_and(x2 >= lambda_min, x2 <= lambda_max)
        y2_masked = y2[mask2]

        if y1_masked.size < 2 > y2_masked.size:
            print('WARNING: compared spectra do not have sufficient overlap. Returning None')
            return None

        if not np.array_equal(x1[mask1], x2[mask2]):
            y2_masked = resample(y2_masked, mask1.sum())

        if use_gradient:
            if is_cube:
                errs = np.gradient(y1_masked, axis=2) - np.gradient(y2_masked)
            else:
                errs = np.gradient(y1_masked) - np.gradient(y2_masked)
        else:
            errs = y1_masked - y2_masked

        if squared_errs:
            errs = np.power(errs, 2)
        else:
            errs = np.abs(errs)

        if is_cube:
            return np.mean(errs, axis=2)
        else:
            return np.mean(errs)
    
    def search_spectrum(self,
                        x_query,
                        y_query,
                        custom_range=None,
                        use_gradient=False,
                        squared_errs=True):
        x_query = np.array(x_query)
        y_query = np.array(y_query)
        is_cube = len(y_query.shape) == 3

        results = []
        v1_memo = {}  # effective_range -> v1 array (per-call memoization)

        for db_spectrum in self.spectra:
            x2 = np.array(db_spectrum.x)
            y2 = np.array(db_spectrum.y)

            # --- Overlap (same math as compare_spectra) ---
            lambda_min = max(x_query[0], x2[0])
            lambda_max = min(x_query[-1], x2[-1])
            if custom_range is not None:
                lambda_min = max(lambda_min, custom_range[0])
                lambda_max = min(lambda_max, custom_range[1])

            mask1 = np.logical_and(x_query >= lambda_min, x_query <= lambda_max)
            mask2 = np.logical_and(x2 >= lambda_min, x2 <= lambda_max)

            # --- Overlap check (same as compare_spectra) ---
            if is_cube:
                y1_masked_size = y_query[:, :, mask1].size
            else:
                y1_masked_size = y_query[mask1].size
            y2_masked_size = y2[mask2].size

            if y1_masked_size < 2 > y2_masked_size:
                continue

            effective_range = (lambda_min, lambda_max)

            # --- v1: memoize by effective_range within this call ---
            if effective_range not in v1_memo:
                v1_memo[effective_range] = self._extractor.extract(
                    x_query, y_query, effective_range, use_gradient)
            v1 = v1_memo[effective_range]

            # --- v2: check VectorStore (persists across calls) ---
            v2_key = self._store.make_db_vector_key(
                x_query, x2, y2, custom_range, use_gradient)
            v2 = self._store.get(v2_key)

            if v2 is None:
                # Resample + extract (same math as compare_spectra)
                if not np.array_equal(x_query[mask1], x2[mask2]):
                    y2 = resample(y2[mask2], mask1.sum())
                    x2 = x_query[mask1]
                v2 = self._extractor.extract(x2, y2, effective_range, use_gradient)
                self._store.put(v2_key, v2)

            # --- Distance (same math as compare_spectra) ---
            errs = v1 - v2
            if squared_errs:
                errs = np.power(errs, 2)
            else:
                errs = np.abs(errs)

            if is_cube:
                error = np.mean(errs, axis=2)
            else:
                error = np.mean(errs)

            results.append({'error': error, 'spectrum': db_spectrum})

        results.sort(key=lambda v: v['error'])
        return results

    @staticmethod
    def export_spectrum(file_spectrum,
                        spectrum,
                        image=None):
        valid = True
        if os.path.splitext(file_spectrum)[1] in ['.dx', '.jdx', '.jcm']:
            spectrum.save_jcamp(file_spectrum)
        elif os.path.splitext(file_spectrum)[1] in ['.dpt', '.txt', '.csv']:
            spectrum.save_dpt(file_spectrum)
        else:
            valid = False
        if valid:
            if image is not None:
                base, ext = os.path.splitext(file_spectrum)
                matplotlib.image.imsave(base + '.png', image)
            return True
        else:
            print('warning: invalid file extension given. spectrum not saved. allowed: '
                  '.dpt, .txt (plain comma-separated x,y values), .dx, .jdx, .jcm (JCAMP-DX)')
            return False


