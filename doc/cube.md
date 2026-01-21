# cube.py — Hyperspectral Cube Loading & Calibration

## Purpose

`cube.py` is responsible for **loading, calibrating, and representing hyperspectral image (HSI) cubes**.

It handles the conversion from instrument-specific file formats into an in-memory data structure that can be used by the rest of Hyperlyse.

This module deliberately contains **no analysis, similarity, or machine-learning logic**.

---

## Conceptual Model

A `Cube` represents **one hyperspectral scan**:

- spatial dimensions: `(rows, cols)`
- spectral dimension: `(bands)`
- one spectrum per pixel

Internally:

```
cube.data[y, x, :]  # spectrum at pixel (x, y)
```

The cube stores **raw, physically meaningful measurements** after radiometric calibration.

---

## Loading and Calibration

When a cube is created:

1. ENVI data and header files are loaded
2. Wavelength information is extracted from metadata
3. Instrument information is read if available
4. Dark and white reference files are applied if present

Calibration is performed as:

- `(data - dark) / (white - dark)` when references exist
- `data / scale_factor` otherwise

This step corrects sensor-specific effects and is **mandatory physics-based calibration**, not preprocessing.

---

## Wavelength Handling

- `self.bands` stores the wavelength (in nm) for each spectral band
- `lambda2layer(...)` maps a wavelength value to the nearest band index

These utilities support UI interaction and wavelength range selection.

---

## Visualization Support

- `to_rgb()` constructs an RGB image using three selected wavelength bands
- RGB output is clipped to `[0, 1]` and intended **only for visualization**

No scientific conclusions should be drawn from RGB composites.

---

## Architectural Role

`cube.py` defines the **input boundary** of Hyperlyse:

```
Instrument files → Cube → Analysis Pipeline
```

It provides:

- calibrated spectral data
- wavelength metadata
- spatial indexing

It does **not** provide:

- spectral preprocessing
- feature extraction
- similarity computation
- dimensionality reduction

---

## Design Constraints

- Calibration logic must remain unchanged
- No preprocessing or ML logic belongs in this module
- Raw spectral values must remain traceable and reproducible

---

## Planned Changes

No changes are planned for `cube.py` in the current refactor.

Future extensions, if any, will be limited to safe helper utilities and will not alter calibration or data semantics.

---

## Summary

`cube.py` encapsulates hyperspectral data loading and radiometric calibration and serves as the stable, authoritative source of spectral measurements for all downstream analysis in Hyperlyse.
