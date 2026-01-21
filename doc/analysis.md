# analysis.py — PCA Utility Functions

## Purpose

`analysis.py` provides **mathematical helper functions** used for exploratory analysis and visualization in Hyperlyse.

It does **not** define similarity, distance metrics, or search behavior.

---

## Current Functionality

### `principal_component_analysis(...)`

This function:

- takes a hyperspectral data cube `(rows × cols × bands)`
- reshapes pixel spectra into samples
- fits a PCA model (optionally on a subset for performance)
- projects all pixels into PCA space
- returns a PCA-transformed cube

The implementation uses:

- scikit-learn PCA
- whitening enabled
- no domain-specific tuning

---

## Role in Hyperlyse Today

- Used **only for visualization** (PCA image display)
- Recomputed on demand
- Not cached or persisted
- Not involved in similarity search or database comparison

---

## Architectural Assessment

- Contains **no algorithmic decision logic**
- Does **not** affect scientific similarity results
- Acts as a reusable numerical transformation utility

---

## Planned Role in New Architecture

- PCA will later be applied to **spectral vectors**, not image cubes
- PCA will become part of the preprocessing pipeline
- The PCA model will be reused for:
  - database vectors
  - query vectors

---

## Design Constraints

- No behavioral change in Phase 1
- Mathematical behavior must remain transparent and reproducible

---

## Summary

`analysis.py` is a **pure ML utility module** that supports dimensionality reduction but does not define or influence spectral similarity semantics.
