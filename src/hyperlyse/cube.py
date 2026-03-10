import os
import numpy as np
import matplotlib.pyplot as plt
import spectral


class Cube:

    DEFAULT_RGB = (598, 548, 449) # wavelengths of red, green, blue, as in the standard settings of SpecimIQ Studio

    def __init__(self, file_data):
        self.data = None
        self.nrows = 0
        self.ncols = 0
        self.nbands = 0
        self.bands = []
        self.rgb_layers = (0, 0, 0)
        self.device = 'unknown device'
        self.__read_data(file_data)


    def __read_data(self, file_data, verbose=False):
        # assemble additional filepaths
        dir_data = os.path.dirname(file_data)
        capture_id, ext = os.path.splitext(os.path.basename(file_data))
        file_header = os.path.join(dir_data, f'{capture_id}.hdr')
        if not os.path.isfile(file_header):
            # try another variant..
            file_header = os.path.join(dir_data, f'{capture_id}{ext}.hdr')

        file_dref_data = os.path.join(dir_data, f'DARKREF_{capture_id}{ext}')
        file_dref_header = os.path.join(dir_data, f'DARKREF_{capture_id}.hdr')
        file_wref_data = os.path.join(dir_data, f'WHITEREF_{capture_id}{ext}')
        file_wref_header = os.path.join(dir_data, f'WHITEREF_{capture_id}.hdr')

        # read main data
        header = spectral.envi.open(file_header, file_data)
        data = header.load()
        self.nrows = data.shape[0]
        self.ncols = data.shape[1]
        self.nbands = data.shape[2]

        # read meta
        self.bands = header.bands.centers
        if 'default bands' in header.metadata:
            self.rgb_layers = (int(header.metadata['default bands'][0]),
                               int(header.metadata['default bands'][1]),
                               int(header.metadata['default bands'][2]))
        else:
            self.rgb_layers = (self.lambda2layer(Cube.DEFAULT_RGB[0]),
                               self.lambda2layer(Cube.DEFAULT_RGB[1]),
                               self.lambda2layer(Cube.DEFAULT_RGB[2]))
        if 'scale factor' in header.metadata:
            scale_factor = float(header.metadata['scale factor'])
        else:
            scale_factor = 1.0
        for device_key in ['sensor type', 'instrument name']:
            if device_key in header.metadata:
                self.device = header.metadata[device_key]
                break

        # read white and black ref
        try:
            dref_header = spectral.envi.open(file_dref_header, file_dref_data)
            dref_data = dref_header.load()
            wref_header = spectral.envi.open(file_wref_header, file_wref_data)
            wref_data = wref_header.load()

            dref_mean = np.mean(dref_data, axis=1)
            wref_mean = np.mean(wref_data, axis=1)

            # plot white and dark references
            if verbose:
                f, (dplot, wplot) = plt.subplots(1, 2)
                dplot.plot(dref_header.bands.centers, dref_mean[0, :])
                dplot.set_title('dark reference')
                wplot.plot(dref_header.bands.centers, wref_mean[0, :])
                wplot.set_title('white reference')
                plt.show()

            # use mean? or use all values? who knows?
            self.data = (data - dref_data) / (wref_data - dref_data)

        except Exception as e:
            #self.data = np.clip(data / scale_factor, 0, 1)
            self.data = data / scale_factor
            print(f"WARNING: Calibration failed ({e}), cube might be uncalibrated.")

        if verbose:
            rgb = self.to_rgb()
            plt.figure(figsize=(10, 10))
            plt.title('Composed RGB image')
            plt.imshow(rgb, extent=(0, 50, 0, 50))
            plt.show()

    def lambda2layer(self, lmd):
        diffs = [abs(lmd-l) for l in self.bands]
        return diffs.index(min(diffs))

    def to_rgb(self):
        rgb = self.data[:,:,self.rgb_layers]
        # clip anything above white
        rgb[rgb > 1] = 1
        return rgb
