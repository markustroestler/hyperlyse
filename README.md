# hyperlyse

Hyperspectral Image Analysis tool

---

See directory <code>doc</code> for documentation and <code>dist_archive</code> for Windows builds.

---

## Running with python

for required python packages, see <code>src/requirements.txt</code>

### Setup Virtual Environment

Create and activate a virtual environment:

**Windows:**

```
python -m venv venv_hyperlyse
venv_hyperlyse\Scripts\activate
```

**Linux/Mac:**

```
python -m venv venv_hyperlyse
source venv_hyperlyse/bin/activate
```

Install dependencies:

```
pip install -r src/requirements.txt
```

### Run the Application

```
cd [repo-root]/src
python main.py
```

---

## Using the Windows buids

1. Download the latest <code>.zip</code> archive from <code>dist_archive</code>
2. unpack
3. run <code>hyperlyse.exe</code>

Known issue: some overly ambitions anti-virus softwares might put some of the necessary files in quarantine when unpacking... then the applications does not start properly.

---

## Making new Windows builds

for required python packages, see requirements.txt

**building with pyinstaller works with Python 3.9 and later (tested with Python 3.11)**

for building with pyinstaller on Windows, cd to /hyperlyse/src/ and run build.bat

---

## Changelog

### v1.3.3

- Data loading: if a "scale factor" is found in ENVI .hdr file, the values of the data cube are scaled accordingly.
  No other transformations/scalings are applied.
- PCA visualizations: only the wavelengths selected sith the wavelength comparison slider are used
  (this is useful, if noisy upper and lower ends of the spectrum lead to "pure noise" components)

### v1.3.2

- bugfix: crash on save similarity image

### v1.3.1 (patch)

- default filename for spetrum export now only contains the object name once

### v1.3

- R009 brightness adjustment slider for visualization image
- R010 UI element for y-range
- configuration in config.json
- semitransparent markers
- R011 define custom spectral range (x-axis) used for all comparison operations (with UI element)
- R012 select spectrum by rectangle (average)
- R013 new database architecture
  - DETAILED REQUIREMENTS
    - load and save, portable db format
    - select, display and compare specific DB spectrum
    - extended fields/metadata for database spectra
    - image of source material
  - SOLUTION
    - based on file system
    - for each spectrum, store:
      - jcamp-dx file with spectrum data and metadata
      - png image with visualization of source
    - load database -> recursivly parse arbitryry folder, or load single file
    - save to database -> just save anywhere
    - delete from database -> delete from filesystem
    - portable -> just copy filesystem
    - new fields and where they are stored in jcamp-dx:
      - name/id ==> ##TITLE
      - origin (e.g. name of color poster or manuscript) ==> ##TITLE
      - original HSI file ==> ##SOURCE REFERENCE
      - rectangle of measurement ==> ##SOURCE REFERENCE
      - measurement device ==> ##SPECTROMETER/DATASYSTEM
      - pigment intensity (light, medium, dark) ==> ##SAMPLE DESCRIPTION
- in the process: lots of internal refactoring

### v1.2

- Migration to Qt6 (also updated the other python packages)
- Zooming: Slider instead of combo box - smaller increments
- R005 higher zoom level by default (--> make windows larger upon startup, make image fill whole area
- R006 no automatic scaling of y-axis (graph)
- R008 support for general envi files (not only from Specim IQ)
- R008.A comparison of spectra with different bands
- R007 PCA - without much user control. might be added on request.

### v1.1

- R001 export image with marked samplepoint together with spectrum
- R002 zoom (for precise point selection)
- R003 export to JCAMP format
- R004 make spectra exports perfectly compatible with exports from SpecimIQ Studio
- spectral databases and spectra comparison features (experimental)
- advanced image view modes

### open feature requests

(none)
