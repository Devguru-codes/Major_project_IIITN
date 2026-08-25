# Major Project (IIITN) — Neurological Disorder Classification Using Signal Processing + Graph Neural Networks

> **Representation Matters:** A Controlled Comparison of EEG Signal Representations and Graph Constructions for Dementia Classification with a Fixed Graph Neural Network

[![Status](https://img.shields.io/badge/stage-research%20%26%20design-blue)]()
[![Decision](https://img.shields.io/badge/decision-GO-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-OpenNeuro%20ds004504-9cf)](https://openneuro.org/datasets/ds004504)
[![Licence](https://img.shields.io/badge/data%20licence-CC0-lightgrey)](https://openneuro.org/datasets/ds004504)
[![Literature](https://img.shields.io/badge/papers%20screened-1091-orange)]()
[![Selected](https://img.shields.io/badge/final%20selection-30%20papers-success)]()

---

## Table of Contents

1. [About This Repository](#about-this-repository)
2. [The Research Question](#the-research-question)
3. [Why This Project Exists (Motivation)](#why-this-project-exists-motivation)
4. [What Is In This Repo (File-by-File Guide)](#what-is-in-this-repo-file-by-file-guide)
5. [Literature Review Methodology](#literature-review-methodology)
6. [Key Findings From the Review](#key-findings-from-the-review)
7. [Final Research Design](#final-research-design)
   - [Disease & Task](#disease--task)
   - [Dataset](#dataset)
   - [Factor A — Six Signal Representations](#factor-a--six-signal-representations)
   - [Factor B — Three Graph Constructions](#factor-b--three-graph-constructions)
   - [The Fixed Model](#the-fixed-model)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Evaluation Protocol & Metrics](#evaluation-protocol--metrics)
   - [Statistical Analysis](#statistical-analysis)
   - [Baselines & Ablations](#baselines--ablations)
8. [Research Questions](#research-questions)
9. [The Research Gap (Documented by Others)](#the-research-gap-documented-by-others)
10. [Three Non-Negotiable Rules](#three-non-negotiable-rules)
11. [Project Timeline](#project-timeline)
12. [Honest Expectations](#honest-expectations)
13. [Target Publication Venues](#target-publication-venues)
14. [Deliverables Checklist](#deliverables-checklist)
15. [Verification Standards Used](#verification-standards-used)
16. [How to Get the Dataset](#how-to-get-the-dataset)
17. [Key References](#key-references)
18. [Repository Status & Roadmap](#repository-status--roadmap)
19. [Author](#author)

---

## About This Repository

This repository contains the **complete research-planning and literature-review phase** of a B.Tech Major Project at the **Indian Institute of Information Technology, Nagpur (IIITN)**.

The project investigates whether **Graph Neural Networks (GNNs)** can improve neurological disorder classification from EEG signals — specifically by asking a question the existing literature has never answered cleanly:

> *When a GNN classifies dementia from EEG, does performance come from **how the signal is represented**, from **how the brain-graph is constructed**, or from the **model architecture** everyone keeps changing?*

Rather than jumping straight to code, this repo documents a rigorous, fully verified research foundation:

- **1,091** unique DOI-bearing works discovered via programmatic OpenAlex queries
- **87** papers examined in full detail
- **30** papers selected for the final literature review
- Every single DOI resolved against **Crossref, DataCite, or doi.org** — nothing cited from memory
- Citation counts cross-checked against **three independent sources** (Crossref, OpenAlex, Semantic Scholar)
- A final **GO decision**, a complete experimental design, and a journal-ready paper blueprint

---

## The Research Question

Published EEG dementia-classification accuracies on **the same public dataset span roughly 80–99%** — an implausibly wide spread for identical data. Studies vary the signal preprocessing, the graph construction, the architecture, and the evaluation protocol **all at once**, making it impossible to know what actually drives performance.

This project **isolates the variables**:

| Variable | Role in our design |
|---|---|
| Signal representation (6 variants) | ✅ **Varied** — Factor A |
| Graph construction / edge definition (3 variants) | ✅ **Varied** — Factor B |
| Model architecture | ❄️ **Frozen** — one fixed 2-layer GCN (~3k params) |
| Dataset | ❄️ **Frozen** — OpenNeuro ds004504 only |
| Evaluation protocol | ❄️ **Frozen** — subject-level nested 5-fold CV × 5 seeds |
| Hyperparameters | ❄️ **Frozen** — identical everywhere, set a priori |

A fully-crossed **6 × 3 factorial** = 18 cells × 5 folds × 5 seeds = **450 runs**, all under one identical pipeline.

Additionally, a dedicated **leakage ablation** re-runs the same pipeline with window-level vs subject-level splitting to quantify exactly how much published numbers are inflated by protocol alone.

---

## Why This Project Exists (Motivation)

Three independent lines of recent evidence motivated this study — none of them authored by us:

1. **The 2026 AHEPA benchmark scoping review** (`10.1007/s11571-026-10464-w`, *Cognitive Neurodynamics*) finds that apparent deep-learning gains on this exact dataset *"arise not from inherent representational advantage, but from data leakage and overfitting to non-independent samples"*, and that added input complexity *"do[es] not demonstrate a systematic performance improvement."*

2. **The closest prior work** (`10.3390/diagnostics16050746`, *Diagnostics* 2026) ran the CNN version of a controlled representation comparison on this dataset and concluded in its own Limitations that it *"does not explicitly model the true spatial topology… alternative graph-based or topology-aware fusion schemes may further enhance… performance."*

3. **An explicit community call** (`10.1038/s41598-025-02018-7`, *Scientific Reports* 2025): *"Future studies should consider employing novel techniques, such as graph convolutional networks… to explore the complex interactions between brain regions."*

**Nobody has crossed representation × graph construction under a fixed model.** That intersection is this project.

---

## What Is In This Repo (File-by-File Guide)

| File | Type | Contents |
|---|---|---|
| [`prompt.docx`](prompt.docx) / [`prompt .pdf`](prompt%20.pdf) | Brief | The original research task specification: objectives, selection criteria, the mandatory "GNN future work" evidence requirement, and reporting rules |
| [`LITERATURE_REVIEW_REPORT.md`](LITERATURE_REVIEW_REPORT.md) | ⭐ Core document | Full **22-section** deep literature review (~84 KB): disease ranking, dataset ranking, search strategy, final 30 papers, 2025–2026 recency cohort, preprocessing taxonomy, GNN recommendation, graph construction, experimental protocol, SOTA benchmarking, research-gap statement, DOI tracking, category assignment, evidence chain |
| [`final_decision.md`](final_decision.md) | ⭐ Decision record | The short version of everything: the GO decision, all locked choices (dataset, task, model, metric, protocol), the week-by-week plan, the three non-negotiable rules, honest expectations, and the deliverables checklist |
| [`paper_structure.md`](paper_structure.md) | Writing blueprint | Section-by-section blueprint for the target journal paper (~7,500–9,000 words): abstract moves, per-section word budgets, figure/table inventory (6 figures, 5 tables), citation map for all 30 refs, recommended writing order, pre-empted reviewer objections, submission checklist |
| [`gnn_evidence.json`](gnn_evidence.json) | Evidence base | **Verbatim quotations** extracted only from full text actually retrieved, classifying papers into Category A (authors explicitly name GNNs as future work), B (recommend graph/topology modelling without saying "GNN"), C (limitation strongly motivates graphs), D (already uses GNNs) |
| [`doi_tracking.csv`](doi_tracking.csv) | Tracking sheet | All **87 examined papers**: ref ID, title, year, venue, DOI + URL, DOI-verified source, citation counts from Crossref/OpenAlex/Semantic Scholar, check date, selected/rejected + rejection reason |
| [`papers_seen_full.csv`](papers_seen_full.csv) | Discovery corpus | The full **1,091-record** screening corpus harvested from OpenAlex (37 structured queries), with open-access flags and discovery-query provenance |
| [`paywalled_papers.md`](paywalled_papers.md) / [`paywalled_papers.csv`](paywalled_papers.csv) | Access report | The **41 papers** whose full text could not be retrieved, split into Group A (11 genuinely paywalled) vs Group B (30 with legal free copies that blocked automated fetching), each with free links where they exist; Unpaywall queried for every entry |
| [`verified_metadata.json`](verified_metadata.json) | Metadata cache | OpenAlex-verified metadata (**79 records**) incl. authors, venue, publisher, citation counts, OA status and abstracts for the tracked corpus |

---

## Literature Review Methodology

### Search strategy (actually executed)

- Programmatic queries against the **OpenAlex API** (`title_and_abstract.search`)
- **37 structured queries** spanning diseases, methods (GNN, GCN, transformer, CNN), pipelines (wavelet, spectrogram, Hjorth, entropy), datasets, and preprocessing comparisons — plus a date-filtered 2025-onward sweep
- Yield: **1,091 unique DOI-bearing works**
- Metadata cross-checked against **Crossref**, **DataCite**, **Semantic Scholar**, and **Europe PMC**

### Screening funnel

```
1,091 discovered (OpenAlex, 37 queries)
      │
     ▼
    87 examined in detail
      │  (full text retrieved for 46; Unpaywall checked for all 41 misses)
     ▼
    30 selected for the final review
       ├── R1–R16 : recent cohort (2025–2026, recency requirement)
       └── F1–F14 : foundational works (datasets, GCN, EEGNet, PLI, leakage framework…)
```

### Selection criterion

Papers were prioritised where the **authors themselves identify graph/GNN-based modelling as future work, a limitation, or an unexplored direction** — with verbatim quotation evidence required. No paper was classified as "GNN as future work" based on inference alone.

### Verification standard

> Every DOI resolved against Crossref (76), DataCite (6, arXiv/OpenNeuro), or HTTP resolution via doi.org. Where a paper is paywalled, no claim is made about its internal content. Where a field could not be verified, it is recorded as **"Not reported"** rather than inferred. Dataset statistics were read directly from the dataset's own files (`participants.tsv`, `eeg.json`, `channels.tsv`, `dataset_description.json`) — not from the dataset paper.

---

## Key Findings From the Review

### Disease ranking (weighted multi-criteria scoring)

| Rank | Disease area | Score | Verdict |
|---|---|---|---|
| 🥇 1 | **Alzheimer's + Frontotemporal dementia (joint)** | **8.10/10** | ✅ Chosen — CC0 dataset, hard unsaturated 3-class problem, well-documented gap |
| 2 | Parkinson's | 7.65 | Retained as external-validation cohort (ds004584) |
| 3 | Epilepsy | 6.70 | Rejected — near performance saturation, crowded GNN literature |
| 4 | MCI (alone) | 6.60 | Rejected — no large open standard-protocol corpus |
| 5–6 | MDD / Schizophrenia | 5.90 / 5.80 | Rejected — access friction + suspiciously high reported accuracies |

### Dataset ranking

| Rank | Dataset | Why / why not |
|---|---|---|
| 🥇 1 | **OpenNeuro ds004504** (AHEPA / Univ. of Ioannina) | Fully open (CC0, no application), single site & protocol, 3-class, de-facto benchmark ✅ |
| 2 | BrainLat | Larger, multimodal — but access-requested and multi-site |
| 3 | CAUEEG | Includes MCI — but access-restricted |
| 4 | OpenNeuro ds004584 (Parkinson's, Iowa) | 149 subjects, CC0 → chosen as external-validation cohort |
| 5–6 | TDBRAIN, ds002778 | Licence restrictions / too small |

---

## Final Research Design

### Disease & Task

- **Task:** 3-class, subject-level classification — **Alzheimer's disease (AD) vs Frontotemporal dementia (FTD) vs Cognitively normal (CN)**
- Binary AD-vs-CN also reported for comparability with prior work
- No dataset merging, ever (see [rules](#three-non-negotiable-rules))

### Dataset

**OpenNeuro ds004504** — *"A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects"* · DOI `10.18112/openneuro.ds004504.v1.0.8` *(v1.0.9 DOI listed in the dataset's own metadata is broken — 404)*

All values below were **read directly from the dataset files**, not the paper:

| Property | Verified value |
|---|---|
| Subjects | **88** — 36 AD / 23 FTD / 29 CN |
| Age (mean ± SD) | AD 66.4±7.9 · FTD 63.7±8.2 · CN 67.9±5.4 |
| MMSE (mean ± SD) | AD 17.8±4.5 · FTD 22.2±2.6 · CN 30.0±0.0 |
| Sex (F/M) | AD 24/12 · FTD 9/14 · CN 11/18 |
| Channels | 19 (10–20 system: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz) |
| Reference | A1–A2 |
| Sampling rate | 500 Hz |
| Condition | Resting state, eyes closed |
| Device | Nihon Kohden EEG 2100 |
| Applied filter | 0.4–50 Hz |
| Total duration | **70,590 s ≈ 19.61 h** · per-subject 307–1291 s (mean 802 s) |
| Format | EEGLAB `.set`, BIDS v1.2.1 |
| Licence | **CC0** — no application, no DUA |

### Factor A — Six Signal Representations

*(what the graph nodes carry)*

| Pipeline | Representation | Node features |
|---|---|---|
| **P1** | Frequency domain — band power + spectral entropy | 19 × 7 |
| **P2** | Connectivity-only — wPLI / PLV / coherence | 19 × 5 |
| **P3** | Time-frequency — CWT / DWT sub-band energy + entropy | 19 × 20 |
| **P4** | Time domain — statistics + Hjorth parameters + entropy | 19 × 10 |
| **P5** | Raw / minimal — z-scored windows | 19 × 1280 |
| **P6** | Fused — P1 + P3 + graph metrics | 19 × ~34 |

### Factor B — Three Graph Constructions

*(how edges are defined)*

| Edge type | Construction |
|---|---|
| **Spatial** | 10–20 electrode coordinates + Gaussian kernel |
| **Functional** | wPLI phase synchronisation |
| **Hybrid** | Element-wise product of both |

Graph properties: kNN k=4 · symmetrised `A = max(A, Aᵀ)` · weighted · undirected · normalised `Â = D̃^(-1/2)(A+I)D̃^(-1/2)`.

### The Fixed Model

- **2-layer GCN** (Kipf & Welling, `10.48550/arXiv.1609.02907`)
- Architecture: 19 → 64 → 32 → 3, mean pooling over nodes
- **~3k parameters** — deliberately small: *the model is the control variable, not the contribution*
- GAT included only as a sensitivity check
- Justification: 2 hops suffice on a 19-node graph; deeper models risk over-smoothing; varying architecture would destroy the comparison
- Training: AdamW, lr 1e-3, weight decay 1e-4, batch 64, ≤200 epochs, early stopping (patience 20, val macro-F1), inverse-frequency class weights

### Preprocessing Pipeline

One identical code path for every condition:

```
band-pass 0.5–45 Hz → 50 Hz notch → average reference
→ ICA (ICLabel; drop eye/muscle components ≥ 0.8)
→ resample to 128 Hz → 10 s non-overlapping windows (≈ 7,059 total)
```

All fitted transforms (scalers, ICA) are fit on the **training fold only**.

### Evaluation Protocol & Metrics

- Subject-level **stratified nested 5-fold CV × 5 seeds** ({0–4}) = **450 runs**
- Fold assignments **frozen** to a committed JSON before any training
- Subject-level prediction = mean of window logits
- **Primary metric: macro-F1** (never plain accuracy — classes are imbalanced)
- Secondary: balanced accuracy, ROC-AUC (OvR macro), PR-AUC, per-class sensitivity/specificity, Cohen's κ, confusion matrices

### Statistical Analysis

- Two-way repeated-measures ANOVA (Representation × Edge definition); interaction term of primary interest
- Wilcoxon signed-rank on paired folds with Holm–Bonferroni correction
- Effect sizes: partial η², Cohen's d; 95% confidence intervals throughout

### Baselines & Ablations

**Baselines:** demographics-only (age + sex — the confound check, run first), LR/SVM/RF on band power, EEGNet, 1D-CNN, published GNN results (with explicit protocol-incomparability caveats).

**Ablations:** k ∈ {3,4,6} · window ∈ {5,10,20} s · weighted vs binarised edges · GCN depth ∈ {1,2,3} · GAT substitution · **leakage ablation** (window-level vs subject-level splitting on an identical pipeline).

---

## Research Questions

1. **RQ1** — Which of six EEG signal representations yields the best macro-F1 for AD/FTD/CN classification under a fixed GCN?
2. **RQ2** — Does graph construction (spatial vs functional vs hybrid) significantly affect performance?
3. **RQ3** — Is there a representation × edge-definition interaction?
4. **RQ4** — Are AD and FTD better separated by node-local spectral properties or by inter-channel connectivity?
5. **RQ5** — How much does window-level (leaky) splitting inflate results relative to subject-level splitting on the identical pipeline?

---

## The Research Gap (Documented by Others)

The gap argument rests on **verbatim third-party evidence** (see `gnn_evidence.json`):

| Cat. | Definition | Papers |
|---|---|---|
| **A** | Authors explicitly name GNNs as future work | R8 (*Sci. Rep.* 2025), F7 (*npj Digital Medicine*) |
| **B** | Authors recommend graph/topology modelling (without saying "GNN") | R2 (*Diagnostics* 2026), F9 (*BMC Neuroscience*), R13 |
| **C** | Stated limitation strongly motivates graph modelling | several |
| **D** | Already uses GNNs (SOTA context, not counted toward gap) | F3, F4, F5, F7, … |

Of 87 examined papers, full text was retrieved for 39 evidence-bearing works; **no Category-A claim rests on paraphrase or on unretrieved text.**

---

## Three Non-Negotiable Rules

1. **Subject-level splits, always.** All windows from one subject stay in one fold; fitted transforms train-fold-only. This is the difference between a real result and a retracted one.
2. **Never feed MMSE to the model.** Controls score MMSE 30.0 ± 0.0 — it is a label in disguise, not a feature.
3. **Never merge datasets to manufacture a multiclass problem.** A combined "Healthy vs AD vs Parkinson's" model would learn site/device/montage — not pathology.

---

## Project Timeline

| Week | Milestone |
|---|---|
| **0** | Sanity check: demographics-only baseline (age + sex). If it predicts well, groups are confounded and nothing else is interpretable |
| 1–2 | Preprocessing module + subject-level splitter + **disjointness unit test** + frozen fold JSON. No training until this passes |
| 3 | Classical baselines (LR/SVM/RF) + EEGNet |
| 4–5 | The 450-run main grid |
| 6 | **Leakage ablation** — best cell re-run with window-level splitting |
| 7 | External validation on ds004584 (does the *ranking* transfer?) |
| 8 | Statistics (RM-ANOVA, Holm-corrected Wilcoxon, effect sizes) + paper writing |

---

## Honest Expectations

- **Target: 80–90% reproducible macro-F1 — not 99%.** Published figures on this dataset run 80–99%, and a 2026 scoping review attributes the top of that range to leakage. Any accuracy above ~95% on 88 subjects with clean splits will be treated as a suspected bug — including our own.
- **Biggest risk:** ~18-subject test folds → wide confidence intervals; small effects may not reach significance.
- **Why we proceed anyway:** given the documented leakage findings, a well-powered *null* result ("representation matters less than assumed, protocol matters more") is itself publishable.
- **Most valuable output:** possibly not the accuracy table, but the leakage-ablation figure — a calibration factor for reading every existing result on this benchmark.

---

## Target Publication Venues

Ranked in `paper_structure.md`:

1. **Journal of Neural Engineering** (IOP) — first choice
2. IEEE JBHI
3. Computers in Biology and Medicine
4. IEEE TNSRE
5. Cognitive Neurodynamics
6. Scientific Reports

Paper spec: ~7,500–9,000 words · 6 figures · 5 tables · 45–60 references.

---

## Deliverables Checklist

- [ ] Preprocessing module with 6 interchangeable representations
- [ ] Subject-level splitter + disjointness unit test
- [ ] Frozen fold assignments committed as JSON
- [ ] Demographics-only confound baseline
- [ ] 450-run grid with full logging
- [ ] Leakage ablation (subject-level vs window-level)
- [ ] ds004584 transfer check
- [ ] Statistics: RM-ANOVA + corrected pairwise tests + effect sizes
- [ ] Public code release: code, seeds, fold assignments, environment
- [ ] Journal manuscript per `paper_structure.md`

---

## Verification Standards Used

- ✅ Every DOI resolved against **Crossref / DataCite / doi.org** — zero DOIs written from memory
- ✅ Citations triple-checked against **Crossref + OpenAlex + Semantic Scholar** (dated 2026-08-17)
- ✅ Dataset facts read from the **dataset's own BIDS files**, independently corroborated against the DICE-Net abstract
- ✅ Category-A "future work" claims require **verbatim quotes from retrieved full text only**
- ✅ Unretrievable papers explicitly flagged; no claims made about their contents
- ✅ Broken v1.0.9 dataset DOI discovered and documented; v1.0.8 used throughout

---

## How to Get the Dataset

```bash
# Direct download (CC0 — no application, no login required)
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504/

# Or via the web browser at:
# https://openneuro.org/datasets/ds004504
```

External-validation cohort (Parkinson's): `aws s3 sync --no-sign-request s3://openneuro.org/ds004584 ds004584/`

Cite as: Miltiadous et al., *Data* 8(6):95, 2023 — DOI [`10.3390/data8060095`](https://doi.org/10.3390/data8060095).

---

## Key References

**Core gap evidence**
- R1 — AHEPA benchmark scoping review, *Cognitive Neurodynamics*, `10.1007/s11571-026-10464-w`
- R2 — Multi-channel EEG time-frequency fusion, *Diagnostics*, `10.3390/diagnostics16050746`
- R8 — EEG connectome dynamics CNN, *Sci. Reports*, `10.1038/s41598-025-02018-7`
- F7 — Interpretable graph learning (Parkinson's EEG), *npj Digital Medicine*, `10.1038/s41746-023-00983-9`
- F8 — Data leakage framework, *Patterns*, `10.1016/j.patter.2023.100804`

**Foundational**
- F13 — Kipf & Welling, GCN, `10.48550/arXiv.1609.02907`
- F14 — EEGNet, `10.1088/1741-2552/aace8c`
- F11 — PLI, `10.1002/hbm.20346` · wPLI, `10.1016/j.neuroimage.2011.01.055`
- F9 — AD ≠ FTD network disturbance (EEG graph theory), `10.1186/1471-2202-10-101`
- F2 — DICE-Net primary benchmark (83.28% LOSO), `10.1109/access.2023.3294618`
- F3 — Closest GNN precedent, *IEEE TNSRE*, `10.1109/tnsre.2022.3204913`
- F5 — GNN-EEG taxonomy survey, `10.1109/tnsre.2024.3355750`

*(Full annotated bibliography of all 30 selected + 57 supporting papers with verified DOIs: see `doi_tracking.csv`.)*

---

## Repository Status & Roadmap

| Phase | Status |
|---|---|
| Research brief received | ✅ Done |
| Deep literature review (1,091 → 87 → 30) | ✅ Done |
| Gap evidence extraction (Categories A–D) | ✅ Done |
| DOI verification & citation triangulation | ✅ Done |
| Experimental design locked (GO decision) | ✅ Done |
| Paper blueprint & venue targeting | ✅ Done |
| Code implementation (preprocessing, splitter, GCN) | 🔜 Next |
| 450-run grid, ablations, external validation | ⏳ Planned |
| Manuscript writing & submission | ⏳ Planned |

---

## Author

**Devguru** ([@Devguru-codes](https://github.com/Devguru-codes))
B.Tech CSE (Batch 2023) · Indian Institute of Information Technology, Nagpur (IIITN)
📧 bt23csd060@iiitn.ac.in

---

## Licence & Attribution

- The EEG dataset (OpenNeuro ds004504) is released under **CC0** by Miltiadous et al. (AHEPA / University of Ioannina).
- Ethics: this study uses publicly released, de-identified secondary data only.
- If you use this repository's literature-review artefacts, please cite the underlying papers (DOIs in `doi_tracking.csv`) rather than this README.
