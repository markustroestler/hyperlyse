# database.py — Spectral Similarity Core

## Purpose

`database.py` contains the **entire algorithmic core** of Hyperlyse.

It defines how:

- spectral reference data is stored,
- spectral similarity is computed,
- similarity search is performed.

All scientific similarity results produced by Hyperlyse ultimately depend on this module.

---

## What This Module Does Today

### Data Structures

- **Metadata**
  Holds provenance and descriptive information (sample ID, source, coordinates, device info).
  This data is **not used for similarity computation** and exists solely for traceability and UI display.

- **Spectrum**
  Represents a **raw spectrum** consisting of:
  - wavelength axis (`x`)
  - intensity values (`y`)
  - associated metadata

  A `Spectrum` is the **authoritative, unprocessed measurement**.

---

### Storage Model

- `Database` loads all spectral reference files (`.jdx`, `.dx`, `.jcm`) from a directory
- Each file is parsed into a `Spectrum` object
- All spectra are stored in memory as **raw data**
- No preprocessing, normalization, or caching is performed

---

## Algorithmic Logic

### `compare_spectra(...)`

This function implements the **full spectral comparison pipeline**:

1. Detects spectrum vs. hyperspectral cube input
2. Determines overlapping wavelength range
3. Applies optional user-defined wavelength limits
4. Masks spectra to the common range
5. Resamples spectra if wavelength grids differ
6. Optionally computes spectral gradients
7. Computes pointwise differences
8. Aggregates differences into a scalar distance

Preprocessing, feature construction, and distance computation are currently **combined in this function**.

---

### `search_spectrum(...)`

Implements a **brute-force similarity search**:

- Iterates over all reference spectra
- Calls `compare_spectra` for each entry
- Sorts results by increasing distance

This is effectively a naïve **K-Nearest-Neighbors (KNN)** search.

---

## Architectural Assessment

- All analytical reasoning in Hyperlyse is located in:
  - `compare_spectra`
  - `search_spectrum`

- All other modules handle UI, visualization, or file I/O
- Spectral preprocessing is implicit and recomputed for every query

---

## Planned Refactor Steps (No Behavior Change)

### Step 1 — Canonical Vector Representation

Introduce a single preprocessing entry point:

```
spectrum_to_vector(...)
```

This function will:

- perform wavelength alignment and masking
- handle resampling and gradient computation
- return a fixed-length numeric vector

---

### Step 2 — Refactor Similarity Computation

- `compare_spectra` will be rewritten to operate on vectors
- Distance computation becomes explicit and reusable
- Numerical results must remain **identical** to the current implementation

---

### Step 3 — Vector Reuse and Caching

- Database spectra will be vectorized once
- Query spectra will be vectorized on demand
- Repeated preprocessing will be eliminated

---

### Step 4 — Preparation for ML Integration

- PCA will be applied to spectral vectors
- KNN search will operate in vector or PCA space
- Cross-cube similarity search becomes possible

---

## Design Constraints

- No UI changes
- No change in scientific interpretation
- `.jdx` files remain the authoritative source
- All transformations must be explicit and reproducible

---

## Summary

`database.py` is the foundation of Hyperlyse’s analytical capabilities.

Refactoring this module enables scalability, performance improvements, machine-learning–based evaluation, and cross-cube spectral search while preserving scientific transparency and existing workflows.
