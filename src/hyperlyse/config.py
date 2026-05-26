import json
import os
class Config:
    def __init__(self,
                 version,
                 config_json):
        """
        Reads config from .json file
        :param version: version string
        :param config_json: .json file containing config data
        """
        self.version = version
        self._config_path = config_json
        with open(config_json, 'r') as f:
            cfg = json.load(f)
        self.default_db_path = cfg['DEFAULT_DB_PATH'] if os.path.isdir(cfg['DEFAULT_DB_PATH']) else os.path.abspath('.')
        self.scroll_speed = cfg['SCROLL_SPEED']
        self.cross_size = cfg['CROSS_SIZE']
        self.marker_colors = cfg['MARKER_COLORS']
        self.marker_alpha = cfg['MARKER_ALPHA']
        self.initial_image_width_ratio = 0.45

        # Phase 3 settings
        self.cube_folder_path = cfg.get('CUBE_FOLDER_PATH', '')
        self.include_subfolders = True
        self.sample_rate = cfg.get('SAMPLE_RATE', 1)
        self.search_in_db = cfg.get('SEARCH_IN_DB', True)
        self.search_in_cubes = cfg.get('SEARCH_IN_CUBES', False)
        self.num_hits = cfg.get('NUM_HITS', 3)
        self.use_pca = cfg.get('USE_PCA', False)

    def save(self):
        """Persist current settings back to config.json."""
        with open(self._config_path, 'r') as f:
            cfg = json.load(f)
        cfg['DEFAULT_DB_PATH'] = self.default_db_path
        cfg['CUBE_FOLDER_PATH'] = self.cube_folder_path
        cfg.pop('INCLUDE_SUBFOLDERS', None)
        cfg['SAMPLE_RATE'] = self.sample_rate
        cfg['SEARCH_IN_DB'] = self.search_in_db
        cfg['SEARCH_IN_CUBES'] = self.search_in_cubes
        cfg['NUM_HITS'] = self.num_hits
        cfg['USE_PCA'] = self.use_pca
        with open(self._config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

    def infotext(self):
        return "\n".join(["Hyperlyse",
                          f"Version: {self.version}",
                          "Author: Simon Brenner",
                          "Institution: TU Wien, Computer Vision Lab",
                          "License: CC-BY-NC-SA"])