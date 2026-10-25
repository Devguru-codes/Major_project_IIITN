# Evaluation Protocol, Not Representation or Topology, Drives Reported EEG Classification of Alzheimer's Disease and Frontotemporal Dementia

B.Tech Major Project, Indian Institute of Information Technology, Nagpur (IIITN)

[![Status](https://img.shields.io/badge/stage-manuscript%20draft%20complete-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-OpenNeuro%20ds004504-9cf)](https://openneuro.org/datasets/ds004504)
[![External](https://img.shields.io/badge/external-OpenNeuro%20ds004584-9cf)](https://openneuro.org/datasets/ds004584)
[![Licence](https://img.shields.io/badge/data%20licence-CC0-lightgrey)](https://openneuro.org/datasets/ds004504)
[![Compute](https://img.shields.io/badge/compute-Kaggle%20CPU%20%2B%200.57%20GPU--h-blue)]()

Published EEG classifiers of Alzheimer's disease (AD), frontotemporal dementia (FTD) and healthy controls on
one public dataset report accuracies from about 50% to 99%. Those studies change the signal representation, the
graph, the model and the evaluation protocol all at once. This project holds the model, the data and the protocol
fixed and varies one thing at a time, to measure what actually moves the number.

The manuscript is in [`paper/`](paper/) ([`main.pdf`](paper/main.pdf)). Every number in it is generated from the
committed runs by [`paper/make_numbers.py`](paper/make_numbers.py); none is typed by hand.

---

## Key results

| Question | Result |
|---|---|
| How much does the evaluation protocol move the score? | Splitting 10-s windows instead of subjects raises subject-level macro-F1 from **0.596 to 0.983** on the identical pipeline (+0.39). |
| How much does the signal representation move it? | Up to **0.13** macro-F1 between the seven representations, descriptively. Once the overlap between cross-validation training sets is accounted for, the difference is **not established** on ds004504 (adjusted p = 0.258; 0.188 on independent folds), but it is in the Parkinson's cohort (p = 0.031). |
| How much does the graph construction move it? | Up to **0.015**. The three graph types are formally equivalent within ±0.05 macro-F1 on average, and a learned graph does no better. |
| Does the graph network beat non-graph models? | No significant difference from logistic regression, SVM or random forest on the same features, nor from EEGNet or a 1D-CNN on raw windows. |
| Best configuration | Wavelet features + spatial graph: macro-F1 **0.596** on independent confirmation folds (95% CI 0.487–0.706, corrected for overlapping training sets), above chance (permutation p = 0.010) and above an age-and-sex baseline (0.411). |
| Can uncorrected tests be trusted? | No. In simulation they declare differences between equally informative representations in **34–80%** of datasets; the adjusted test in 0%. With 88 subjects, only gaps above about **0.19** macro-F1 are detectable. |
| PSWE marker | Paroxysmal slow-wave events, a blood–brain-barrier-dysfunction-associated EEG marker, are more frequent in AD (rate ratio 2.96) and FTD (2.07), but not in Parkinson's disease. The excess tracks background slowing (ρ = 0.88), so fixed-threshold PSWEs on scalp EEG mainly index slowing. |

The dataset has no blood–brain-barrier imaging: PSWEs are treated strictly as an associated marker, never as a
measurement of barrier integrity.

---

## Study design

| Element | Choice |
|---|---|
| Data | OpenNeuro **ds004504**: resting, eyes-closed EEG, 19 channels, 88 subjects (36 AD, 23 FTD, 29 controls), 7,013 ten-second windows. External cohort: **ds004584**, Parkinson's disease (100) vs controls (49). |
| Preprocessing (fixed) | 0.5–45 Hz band-pass, mains notch, average reference, ICA with ICLabel (eye/muscle components with p ≥ 0.8 removed), resample to 128 Hz, 10-s windows. |
| Factor A: 7 node-feature representations | P1 spectral (7) · P2 wPLI connectivity (5) · P3 wavelet (20) · P4 time-domain (10) · P5 raw window (1,280) · P6 fused (34) · P7 PSWE-derived (4) |
| Factor B: 3 graph constructions | Spatial (electrode distance) · functional (alpha-band wPLI) · hybrid (their product); kNN k = 4, weighted, GCN-normalised |
| Fixed model | Two-layer GCN, 19 → 64 → 32 → 3, about 3k parameters, hyperparameters fixed before any result |
| Protocol | Subject-level stratified 5-fold CV × 5 repeats (25 frozen splits), plus 25 independent confirmation splits; subject prediction = mean of window logits; primary metric macro-F1 |
| Statistics | Primary: split-level repeated-measures ANOVA with the Nadeau–Bengio adjustment for overlapping training sets, calibrated by simulation. Also: equivalence tests (±0.05), out-of-bootstrap comparison, permutation test, Holm correction throughout. |
| Checks | Matched non-graph baselines, EEGNet and 1D-CNN, 16 ablations, leakage experiment, stopping-rule sensitivity, learned adjacency, occlusion importance, external validation |

### Design rules

1. **Subject-level splits, always.** All windows of a subject stay in one fold, and every cross-subject transform is fitted on training subjects only. Fold files are frozen and asserted disjoint on every load.
2. **MMSE is never a model input.** Every control scores 30, so it is a proxy for the label.
3. **Datasets are never merged.** The Parkinson's cohort is analysed separately, as an external test of the protocol.

---

## Repository layout

```
src/eegrep/        the package: download, preprocessing, features, graphs, model, training, runner, statistics, figures
  data/            OpenNeuro downloader (pinned dataset version) and participants table
  cli.py           every step as a command: python -m eegrep <command>
configs/           default.yaml (all frozen settings), ds004584.yaml, ablation configs
splits/            frozen fold assignments (selection, confirmation, external)
tests/             pytest suite (fold disjointness, no MMSE, scaler fitting, wPLI, PSWE detector, statistics, ...)
kaggle/            generated Kaggle notebooks (NB00-NB26) and build_notebooks.py that writes them
reports/           every raw run (one JSON line per run) and every analysis output, as downloaded from Kaggle
paper/             manuscript (main.tex, sections/), make_numbers.py, generated numbers/tables/figures, main.pdf
IMPLEMENTATION_PLAN.md   pre-specified plan (committed before any data were analysed) and dated log of every change
```

Planning-phase documents from before the implementation are kept at the top level:
[`LITERATURE_REVIEW_REPORT.md`](LITERATURE_REVIEW_REPORT.md) (22-section literature review),
[`final_decision.md`](final_decision.md) (GO decision and locked design),
[`paper_structure.md`](paper_structure.md) (original paper blueprint),
[`gnn_evidence.json`](gnn_evidence.json) (verbatim future-work evidence),
[`doi_tracking.csv`](doi_tracking.csv) (verified DOIs and citation counts),
[`papers_seen_full.csv`](papers_seen_full.csv) (1,091-record screening corpus),
[`paywalled_papers.md`](paywalled_papers.md) / [`.csv`](paywalled_papers.csv) and
[`verified_metadata.json`](verified_metadata.json).
The review screened 1,091 works, examined 87 and selected 30; every DOI was resolved against Crossref or DataCite.

---

## Reproducing the study

All computation ran on free Kaggle sessions, on CPU except the two deep-learning baselines (0.57 GPU-hours).
The local machine only edits code and builds the paper.

**1. Install** (Python ≥ 3.10; PyTorch is preinstalled on Kaggle and not listed as a dependency):

```bash
pip install -e ".[preprocess,stats,dev]"
```

**2. Run the tests:**

```bash
python -m pytest -q
```

**3. Get the data.** The downloader reads the public OpenNeuro bucket over HTTPS and refuses any dataset version
other than the one the study used:

```bash
python -m eegrep download --dest data/ds004504
```

**4. Run the pipeline on Kaggle.** Each step is an `eegrep` command, and the notebooks in [`kaggle/`](kaggle/)
chain them (preprocess → features and graphs → frozen folds → grid → ablations → statistics). See
[`kaggle/README.md`](kaggle/README.md).

**5. Rebuild the paper** from the committed results:

```bash
cd paper && python make_numbers.py && latexmk -pdf main.tex
```

Numbers are reproduced exactly from the committed runs. Re-running from the raw recordings gives approximately the
same numbers: a feature cache rebuilt in a later Kaggle environment shifts individual scores slightly (mean
absolute change 0.065 per split for the best configuration), and the conclusions are unchanged.

---

## Status

| Phase | Status |
|---|---|
| Literature review and design | ✅ Done |
| Implementation, tests, Kaggle pipeline | ✅ Done |
| Main grid, confirmation folds, baselines, ablations, external cohort, PSWE analysis | ✅ Done |
| Statistical calibration and robustness checks | ✅ Done |
| Manuscript draft (target: *Journal of Neural Engineering*) | ✅ Done |
| Author statements, journal template, submission | 🔜 Next |

---

## Data and citation

- ds004504: Miltiadous et al., *Data* 8(6):95, 2023, [`10.3390/data8060095`](https://doi.org/10.3390/data8060095); dataset DOI `10.18112/openneuro.ds004504.v1.0.8` (CC0).
- ds004584: OpenNeuro, DOI `10.18112/openneuro.ds004584.v1.0.0`.
- This study is a secondary analysis of public, de-identified data.

## Author

**Devguru** ([@Devguru-codes](https://github.com/Devguru-codes)) · B.Tech CSE (Batch 2023), IIIT Nagpur · bt23csd060@iiitn.ac.in
