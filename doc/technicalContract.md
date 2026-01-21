# Technical Interpretation & Implementation Contract  
**Project:** Hyperlyse – Scalable Spectral Similarity & ML-Based Evaluation  
**Purpose:** Translate the scientific proposal into concrete, verifiable technical requirements

---

## 1. Scope & Intent

This document translates the scientific project proposal into **software and system requirements**.

The goal is **not** to change the user workflow or scientific intent, but to:
- make the existing functionality scalable,
- introduce a clear machine-learning–based evaluation layer,
- and enable efficient large-scale similarity analysis across many spectra and images.

This document defines **what will be implemented**, **what will not be implemented**, and **how success is measured**.

---

## 2. What Already Exists (Baseline)

The current Hyperlyse system already supports:

- Saving reference spectra (`.jdx`) from pixel or area selections
- Comparing a selected spectrum against:
  - all pixels of the current HSI cube (“current cube selection”)
  - all spectra stored in a database folder (“current database selection”)
- Visual similarity maps and ranked similarity results
- Manual wavelength range selection and threshold (`t`) filtering

These capabilities **must remain functionally unchanged from a user perspective**.

---

## 3. Core Problems Identified

The current implementation has the following limitations:

1. **No reuse of processed spectra**  
   - All database spectra are reloaded and reprocessed on every new selection.
   - This leads to unnecessary recomputation and poor scalability.

2. **No explicit feature representation**  
   - Spectral preprocessing, alignment, and comparison are intertwined.
   - There is no canonical “vector” representation of spectra.

3. **Limited robustness for mixtures and subtle variations**  
   - Raw-spectrum comparison is sensitive to noise and degradation.
   - Mixtures and low-concentration pigments are often missed.

4. **No scalable cross-cube comparison**  
   - Comparison is limited to:
     - current cube vs database spectra
   - There is no unified search space across many HSI cubes.

5. **No measurable performance guarantees**  
   - Time savings are not demonstrable or reproducible.

---

## 4. Functional Requirements (What Will Be Built)

### FR1 – Canonical Spectrum Representation
A single, reusable preprocessing pipeline shall be introduced:

- A function `spectrum_to_vector(...)` will:
  - take raw spectral data as input,
  - apply wavelength selection and alignment,
  - return a fixed-length numeric vector.

This function becomes the **only** entry point for spectral comparison.

---

### FR2 – Vector Reuse & Caching
- Reference spectra stored as `.jdx` files shall be:
  - preprocessed **once** into vectors,
  - cached in memory and/or on disk,
  - reused for all subsequent comparisons.

- For each user selection:
  - only the **query vector** is recalculated.

---

### FR3 – Proper KNN-Style Similarity Search
- Similarity search shall be formalized as a **K-Nearest-Neighbors (KNN)** process:
  - query vector vs a set of precomputed vectors,
  - ranked by distance.

- The initial implementation may be brute-force KNN.
- Results must be **numerically equivalent** to the current system before further optimization.

---

### FR4 – Machine Learning–Based Evaluation (Minimal & Classical)
To satisfy the “machine learning–based evaluation” requirement:

- **PCA (Principal Component Analysis)** shall be introduced:
  - trained on existing spectral vectors,
  - used to reduce dimensionality before KNN.

This is considered sufficient ML for this project:
- no deep learning,
- no supervised classification,
- no black-box models.

---

### FR5 – Cross-Cube / Multi-Image Comparison
The system shall support:

- comparing a selected spectrum against:
  - spectra from multiple HSI cubes,
  - potentially at pixel level.

This enables:
- manuscript-to-manuscript comparison,
- large-scale pattern detection across collections.

---

### FR6 – Performance Improvement & Time Savings
The system shall demonstrate:

- measurable reduction in evaluation time,
- especially for repeated similarity searches.

Performance improvements must be shown via:
- before/after benchmarks,
- identical results with reduced computation time.

---

## 5. Non-Functional Requirements

### NFR1 – Scientific Reproducibility
- Original `.jdx` files remain the **authoritative data source**.
- Vector representations are derived, disposable artifacts.
- Changing preprocessing parameters must invalidate cached vectors.

---

### NFR2 – No Workflow or UI Changes
- Existing UI behavior remains unchanged:
  - same buttons,
  - same sliders,
  - same similarity visualization semantics.
- The meaning of threshold `t` remains:
  - lower `t` = stricter similarity,
  - higher `t` = looser similarity.

---

### NFR3 – Transparency
- All processing steps must be explicit and inspectable.
- No hidden or implicit transformations.

---

## 6. Explicit Non-Goals (What Will NOT Be Done)

The project explicitly does **not** include:

- Deep learning or neural networks
- Automatic pigment classification
- Replacement of `.jdx` with proprietary formats
- UI redesign
- Real-time guarantees
- GPU or cluster-based computation (unless later required)

---

## 7. Acceptance Criteria (Definition of Done)

The project is considered complete when:

1. Database spectra are no longer recomputed on every selection.
2. All comparisons use a shared `spectrum_to_vector` pipeline.
3. PCA-before-KNN is implemented and documented.
4. Cross-cube similarity search is technically possible.
5. Existing results remain reproducible and interpretable.

---

## 8. Summary Statement

> The project does not introduce new scientific claims.  
> It transforms an existing research prototype into a scalable, reproducible, and ML-augmented analysis system suitable for large manuscript collections.

---

**This document represents the technical contract for implementation.**
