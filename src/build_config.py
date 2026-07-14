"""Generate a shipping config.json for the packaged build.

The distributed app should not carry the developer's local machine paths, so this
writes a sanitized copy of config.json with the default paths blanked out into a
temp folder. The build scripts bundle that copy and delete the folder afterwards.
The developer's own src/config.json (used by `python main.py`) is left untouched.
"""
import json
import os

SRC = 'config.json'
OUT_DIR = '_dist_config'
BLANK_KEYS = ('DEFAULT_DB_PATH', 'CUBE_FOLDER_PATH')

with open(SRC, 'r') as f:
    cfg = json.load(f)

for key in BLANK_KEYS:
    cfg[key] = ''

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'config.json')
with open(out_path, 'w') as f:
    json.dump(cfg, f, indent=2)

print(f'build_config: wrote {out_path} (blanked: {", ".join(BLANK_KEYS)})')
