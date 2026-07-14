@echo off
REM Generate a shipping config.json with blanked default paths (dev config.json stays untouched)
python build_config.py
if errorlevel 1 (
    echo build_config.py failed - aborting build.
    exit /b 1
)

pyinstaller --noconfirm --noconsole --icon=icon.ico --hidden-import="sklearn.utils._typedefs" --add-data "_dist_config/config.json;." --add-data "startup.png;." --add-data "icons;icons" --distpath "../dist" --workpath "../build" --name hyperlyse main.py

REM Clean up the temporary build config
rmdir /s /q _dist_config
