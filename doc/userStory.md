# New User Story & System Architecture  
**Project:** Hyperlyse – Cross-Cube Spectral Similarity Search  
**Purpose:** Define the new analysis workflow enabled by the project

---

## 1. Motivation

The current Hyperlyse workflow allows users to:

- store reference spectra (“fingerprints”) from selected areas
- search for these fingerprints inside a single hyperspectral image (HSI cube)

However, research questions increasingly require **comparative analysis across many manuscripts**, such as:

- Does a specific pigment appear in multiple manuscripts?
- In which other scans does a selected material occur?
- Where and how often does a spectral pattern reappear across collections?

To support these questions, a new workflow and system architecture is required.

---

## 2. High-Level User Story (New Capability)

### New user question enabled by the system:

> **“I have selected a material in this manuscript.  
Where else does this same material occur in other analyzed manuscripts, and where?”**

This inverts the current workflow from:
- *reference → single image*

to:
- *selected spectrum → many images*

---

## 3. Conceptual Shift in Architecture

### From:
- isolated analysis of one HSI cube at a time
- small, manually curated reference databases

### To:
- a **shared spectral search space**
- where spectra from many scans are represented uniformly
- and can be queried in one step

---

## 4. Core Concepts (Unified Mental Model)

### Hyperspectral cube
- One scan of one object (e.g. one manuscript page)
- Contains ~250,000 pixels
- Each pixel has one spectrum (across ~200–300 wavelengths)

### Spectrum
- The material fingerprint of one pixel or selected area
- Represented as a numeric vector after preprocessing

### Spectral index (conceptual)
- A collection of many preprocessed spectra
- Each spectrum is linked to:
  - the scan (cube) it originates from
  - its spatial location in that scan

This index is not necessarily a traditional database, but a reusable search structure.

---

## 5. Two-Phase Workflow

### Phase A — Batch Analysis (Preparation)

This phase is typically performed once per dataset or collection.

**Purpose:**
- Convert many hyperspectral scans into a searchable spectral representation.

**Process:**
1. Select a set of HSI scans (cubes)
2. For each scan:
   - extract spectra (per pixel or selected regions)
   - clean and normalize spectral data
   - convert spectra into fixed-length vectors
   - project vectors into a compressed feature space (e.g. PCA)
3. Store:
   - compressed vectors
   - references to scan files and locations
   - preprocessing metadata

**Outcome:**
- A prepared spectral search space covering many scans
- No repeated preprocessing during interactive use

This step is not part of the normal interactive UI workflow.

---

### Phase B — Interactive Analysis (User Workflow)

This phase is identical in look and feel to the existing Hyperlyse workflow.

**User actions:**
1. Open a hyperspectral image in Hyperlyse
2. Select a pixel or area of interest
3. Trigger similarity search

**System behavior:**
1. Convert the selected spectrum into a query vector
2. Compare this vector against all stored spectral vectors
3. Identify:
   - which scans contain similar spectra
   - where in each scan these occur
4. Present results visually and/or as ranked lists

**Key property:**
- Only the query spectrum is processed at interaction time
- All reference data is reused

---

## 6. What the User Gains

With this workflow, users can:

- Find occurrences of a material across many manuscripts
- Detect shared pigment usage between collections
- Identify regional or workshop-specific material patterns
- Reduce manual, repetitive analysis
- Perform large-scale comparisons that were previously impractical

---

## 7. What Does Not Change

- The existing UI interaction model
- The meaning of similarity thresholds
- The use of open spectral formats (e.g. `.jdx`)
- The scientific interpretation of results

The system extends existing capabilities rather than replacing them.

---

## 8. Design Principles

- **Reproducibility:** Raw spectral data remains untouched and traceable
- **Transparency:** All preprocessing steps are explicit and documented
- **Scalability:** Preprocessing is separated from interactive analysis
- **Minimal ML:** Classical, explainable methods are used (e.g. PCA, KNN)

---

## 9. Summary Statement

> The new architecture transforms Hyperlyse from a single-image similarity tool into a multi-manuscript spectral search system, enabling users to locate and compare material fingerprints across large collections efficiently and reproducibly.

---

**This document defines the new user story and system behavior introduced by the project.**
