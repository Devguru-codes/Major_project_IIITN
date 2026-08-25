# Final Decision — What We Are Building

**Date:** 17 August 2026 · **Status:** GO
**Full reasoning:** `LITERATURE_REVIEW_REPORT.md` (22 sections). This file is the short version.

---

## The one-sentence version

We will run a controlled experiment that answers **"does the way you represent an EEG signal change how well a GNN classifies dementia?"** — by holding the dataset, the splits, and the graph model completely fixed and varying only the representation.

---

## The decisions

| | Decision |
|---|---|
| **Disease** | Alzheimer's + frontotemporal dementia + healthy controls (scored 8.10/10; Parkinson's 7.65 second) |
| **Dataset** | OpenNeuro **ds004504** (AHEPA / Univ. of Ioannina) |
| **Dataset DOI** | `10.18112/openneuro.ds004504.v1.0.8` ⚠ *not v1.0.9 — that DOI is broken (404)* |
| **Dataset paper** | `10.3390/data8060095` |
| **Task** | **3-class, subject-level: AD vs FTD vs CN** (binary AD vs CN also reported, for comparability) |
| **Model** | **2-layer GCN**, ~3k params, identical in every run. GAT only as a sensitivity check |
| **Graph** | Nodes = 19 channels · Edges = wPLI × spatial adjacency · weighted, undirected, kNN k=4 |
| **Primary metric** | **Macro-F1** (never plain accuracy — the classes are imbalanced) |
| **Validation** | Subject-level nested 5-fold CV × 5 seeds |
| **External check** | OpenNeuro **ds004584** (100 PD / 49 control) — does the *ranking* transfer? |

**The data, verified directly from the dataset files (not from the paper):**
88 subjects — **36 AD / 23 FTD / 29 CN** · 19 channels (10–20) · 500 Hz · eyes-closed resting · **19.61 hours** total · **CC0 licence, no application needed**.

---

## The experiment

A **6 × 3 fully-crossed factorial** = 18 cells × 5 folds × 5 seeds = **450 runs**.
Each run is a ~3k-parameter model — this is hours of GPU time, not weeks.

**Factor A — the representation (what the nodes carry):**

| | Pipeline | Node features |
|---|---|---|
| P1 | Frequency-domain band power + spectral entropy | 19 × 7 |
| P2 | Connectivity-only (wPLI / PLV / coherence) | 19 × 5 |
| P3 | Time-frequency (CWT / DWT sub-band energy + entropy) | 19 × 20 |
| P4 | Time-domain statistics + Hjorth + entropy | 19 × 10 |
| P5 | Raw / minimal (z-scored windows) | 19 × 1280 |
| P6 | Fused (P1 + P3 + graph metrics) | 19 × ~34 |

**Factor B — the edges (how the graph is built):** spatial · functional (wPLI) · hybrid

Everything else is frozen: identical filtering (0.5–45 Hz, 50 Hz notch), identical ICA, identical 10 s windows at 128 Hz, identical folds, identical optimiser and hyperparameters.

---

## Why this is publishable

The gap is documented by other people, not just by us:

- A **2026 scoping review** of this exact dataset (`10.1007/s11571-026-10464-w`) finds that apparent deep-learning gains *"arise not from inherent representational advantage, but from data leakage and overfitting to non-independent samples"* — and that added representations *"do not demonstrate a systematic performance improvement."*
- A **2026 Diagnostics paper** (`10.3390/diagnostics16050746`) ran the CNN version of our study and concluded in its Limitations that it *"does not explicitly model the true spatial topology… alternative **graph-based** or topology-aware fusion schemes may further enhance… performance."*
- A **2025 Scientific Reports paper** (`10.1038/s41598-025-02018-7`) states outright: *"Future studies should consider employing novel techniques, such as **graph convolutional networks**… to explore the complex interactions between brain regions."*

**Nobody has crossed representation with graph construction under a fixed model.** That intersection is ours.

---

## Order of work

| Week | Do this |
|---|---|
| **0** | **Sanity check first.** Train a classifier on age + sex alone. If it scores well, the groups are demographically confounded and nothing else is interpretable. One day. |
| 1–2 | Build the preprocessing module + the splitter. **Unit-test that train/test subject IDs never overlap.** Freeze folds to a committed JSON. No training until this passes. |
| 3 | Classical baselines (LR / SVM / RF on band power) + EEGNet. |
| 4–5 | The 450-run main grid. |
| 6 | **The money ablation:** re-run the best cell with *window-level* splitting instead of subject-level, and report both numbers. |
| 7 | External validation on ds004584. |
| 8 | Statistics (two-way RM-ANOVA, Holm-corrected Wilcoxon, effect sizes) + writing. |

---

## Three rules we do not break

1. **Subject-level splits, always.** All windows from one person stay in one fold. Scalers and ICA fit on the training fold only. This is the difference between a real result and a retracted one.
2. **Never feed MMSE to the model.** Controls are MMSE 30.0 ± 0.0 — it is a label in disguise, not a feature.
3. **Never merge datasets to make a multiclass problem.** A "Healthy vs AD vs Parkinson's" model built from ds004504 + ds004584 would learn site, device, montage and eyes-open/closed — not pathology. It would look excellent and mean nothing.

---

## What we are aiming for, honestly

**80–90% macro-F1, reproducible** — not 99%.

Published figures on this dataset run from 80% to 99%, and the 2026 review attributes the top of that range to leakage. Any result above ~95% on 88 subjects with proper subject-level splits should be treated as a bug, **including our own**.

**Biggest risk:** 88 subjects means test folds of only ~18 people, so confidence intervals will be wide and small effects may not reach significance.
**Why we proceed anyway:** given the leakage finding above, a well-powered *null* result — "representation matters less than people assume, and protocol matters more" — is itself a publishable contribution. The study is informative whichever way it lands.

**The single most valuable output** may not be the accuracy table at all. It is the week-6 figure: same pipeline, same model, same data, two splitting protocols, side by side. That gives the field a calibration factor for reading every existing ds004504 result.

---

## Deliverables checklist

- [ ] Preprocessing module with 6 interchangeable representations
- [ ] Subject-level splitter + the disjointness unit test
- [ ] Frozen fold assignments committed as JSON
- [ ] Demographics-only baseline (do this first)
- [ ] 450-run grid with full logging
- [ ] Leakage ablation (subject-level vs window-level)
- [ ] ds004584 transfer check
- [ ] Stats: RM-ANOVA + corrected pairwise + effect sizes
- [ ] Public repo: code, seeds, folds, environment

---

### Companion files

| File | Contents |
|---|---|
| `LITERATURE_REVIEW_REPORT.md` | Full 22-section review, final 30 papers, rankings, protocol |
| `gnn_evidence.json` | Verbatim GNN future-work quotes with sources |
| `doi_tracking.csv` | All 87 examined papers, verified DOIs, 3 citation sources |
| `papers_seen_full.csv` | Full 1,091-record discovery corpus |
| `paywalled_papers.md` | The 41 papers without retrieved full text, with free links where they exist |
