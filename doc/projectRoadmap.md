# Project Roadmap — Hyperlyse Scalable Spectral Search

**Goal:** Extend Hyperlyse with scalable, ML-based, cross-cube spectral similarity search  
**Scope:** Software architecture, preprocessing, ML integration, performance improvements

---

## Phase 0 — Alignment & Baseline

### Objective

Ensure shared understanding, stable baseline, and measurable starting point.

### Tasks

- [ ] Confirm scope and responsibilities with project leader
- [ ] Freeze current Hyperlyse version as baseline
- [ ] Run and document current similarity workflow
- [ ] Measure baseline performance:
  - time per similarity search
  - recomputation behavior
- [ ] Collect 3–5 representative HSI cubes for testing

### Deliverables

- Baseline performance notes
- Confirmed scope agreement
- No code changes yet

---

## Phase 1 — Code Understanding & Refactor Preparation

### Objective

Create safe refactor seams without changing behavior.

### Tasks

- [ ] Analyze `database.py` comparison pipeline
- [ ] Identify all places where:
  - spectra are loaded
  - wavelength filtering happens
  - distances are computed
- [ ] Design `spectrum_to_vector(...)` API
- [ ] Add **no-op implementation**:
  - same logic as before
  - just centralized

### Rules

- No behavioral change
- Results must be identical to baseline

### Deliverables

- `spectrum_to_vector` function exists
- All comparisons go through it
- Tests / manual checks show identical output

---

## Phase 2 — Vector Caching & Reuse

### Objective

Eliminate repeated preprocessing of database spectra.

### Tasks

- [ ] Introduce vector cache for database spectra
- [ ] Compute vectors once per `.jdx` entry
- [ ] Store vectors in memory (initially)
- [ ] Ensure:
  - query vector is recomputed
  - database vectors are reused
- [ ] Validate numerical equivalence to baseline

### Deliverables

- Cached database vectors
- Measurable speedup for repeated queries
- No UI changes

---

## Phase 3 — Persistent Vector Storage

### Objective

Avoid recomputation across restarts.

### Tasks

- [ ] Design on-disk vector format (`.npz` + metadata)
- [ ] Store:
  - vector
  - cube/source reference
  - preprocessing version hash
- [ ] Load vectors at startup or on demand
- [ ] Implement cache invalidation if parameters change

### Deliverables

- Persistent vector store
- Startup no longer recomputes vectors
- Reproducible preprocessing

---

## Phase 4 — ML Integration: PCA Before Similarity

### Objective

Improve robustness to mixtures and noise using classical ML.

### Tasks

- [ ] Train PCA on existing spectral vectors
- [ ] Choose PCA dimensionality (e.g. 10–30)
- [ ] Apply PCA transform to:
  - stored vectors
  - query vectors
- [ ] Run KNN / distance search in PCA space
- [ ] Compare results with baseline:
  - fewer false negatives
  - stable behavior

### Deliverables

- PCA model saved & versioned
- PCA-before-KNN pipeline active
- Documented improvement rationale

---

## Phase 5 — Cross-Cube Search Capability

### Objective

Enable “search this spectrum across many scans”.

### Tasks

- [ ] Extend vector store to include:
  - cube ID
  - pixel / region location
- [ ] Implement batch preprocessing for multiple cubes
- [ ] Enable query against:
  - vectors from cube A
  - cube B
  - cube C
- [ ] Return:
  - list of cubes with matches
  - approximate locations

### Deliverables

- Cross-cube similarity search
- Inverted query supported
- No change to existing UI interaction

---

## Phase 6 — Performance Validation & Benchmarks

### Objective

Demonstrate time savings and scalability.

### Tasks

- [ ] Benchmark:
  - baseline vs new pipeline
  - single cube
  - multi-cube search
- [ ] Measure:
  - preprocessing time
  - query time
  - memory usage
- [ ] Document results clearly

### Deliverables

- Before/after performance comparison
- Evidence of time savings
- Benchmark notes for reporting

---

## Phase 7 — Documentation & Handover

### Objective

Make the system understandable, defensible, and reusable.

### Tasks

- [ ] Document:
  - preprocessing pipeline
  - vector format
  - PCA usage
- [ ] Write developer notes:
  - how to add new scans
  - how to rebuild the index
- [ ] Prepare explanation for:
  - “machine learning–based evaluation”
  - reviewers and collaborators

### Deliverables

- Technical documentation
- User-story documentation
- Clean handover state

---

## Definition of Done

The project is complete when:

- [ ] Database spectra are not recomputed per query
- [ ] `spectrum_to_vector` is the single preprocessing entry point
- [ ] PCA + KNN are active and documented
- [ ] Cross-cube similarity search is possible
- [ ] Time savings are measurable and reproducible
- [ ] No UI or scientific workflows are broken

---

## Final Statement

> This roadmap delivers the promised ML-based, scalable evaluation of hyperspectral data while preserving scientific transparency and existing user workflows.

---
