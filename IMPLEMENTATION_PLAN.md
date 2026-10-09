# End-to-End Plan: EEG Representation × Graph × BBB-Linked Slow-Wave Marker (fixed GCN, ds004504, all on Kaggle)

## Context

Repo `Devguru-codes/Major_project_IIITN` (local clone: `C:\Users\devgu\Downloads\research_paper_for_brain_related\Major_project_IIITN`) holds only the finished planning and literature phase:
- `final_decision.md`: GO decision and locked design
- `LITERATURE_REVIEW_REPORT.md`: pipeline specs in §8–§13
- `paper_structure.md`: the paper blueprint
- the evidence CSV and JSON files

There is no code yet. You asked for the following.
- **Plan only for now.** Commits go on a local branch.
- **All compute on Kaggle.** That includes CPU work, and GPU hours must stay small.
- **A brain-related paper with a blood–brain-barrier (BBB) angle.** It must be novel and of real quality, with justified ablations, multiple seeds, real statistics, confusion matrices and a full range of figures.

**BBB decision (yours): EEG + PSWE arm.** Paroxysmal slow-wave events (PSWEs) are stretches where the EEG median power frequency stays below 6 Hz for at least 5 s. Milikovsky et al. (*Sci. Transl. Med.* 2019, `10.1126/scitranslmed.aaw8954`) reported them in AD patients, with rate correlating with cognitive impairment. In epilepsy patients the events localised to cortex with BBB dysfunction, and albumin infusion caused them in rats, so causation is shown only in animals. My searches found no PSWE study in FTD and none on ds004504. ds004504 has no BBB imaging, so the paper says **"BBB-dysfunction-associated EEG marker" throughout and never "BBB measurement."**

**Working title:** *Representation, Topology, and a Blood–Brain-Barrier-Linked Slow-Wave Marker: A Controlled Graph Neural Network Study of EEG in Alzheimer's Disease and Frontotemporal Dementia*

---

## 1. Novelty and contributions (what makes this publishable)

| # | Contribution | Why it is new | Gap evidence |
|---|---|---|---|
| C1 | Fully-crossed **7 representations × 3 graph constructions** under one fixed GCN on ds004504 | No prior work varies both under a fixed model. Radwan et al. 2025 (bioRxiv `10.1101/2025.08.18.670945`) use ds004504 with one graph model, the authors' preprocessed data and window-level prediction; they are added to Table 1. | R1, R2, R8, F7 (repo `gnn_evidence.json`) |
| C2 | **First PSWE characterisation in FTD vs AD vs CN**, plus an independent open-cohort replication of the AD finding | The 2019 STM paper covered AD and epilepsy, a 2023 Sensors paper covered PD, and a 2025 EMBC paper covered epilepsy. None covered FTD or ds004504. | Milikovsky 2019; s23020918 (PD); EMBC 2025 |
| C3 | **Incremental-value test**: does the BBB-linked PSWE marker add information beyond spectral representations inside a GNN? It also introduces a **PSWE co-occurrence graph**. | No prior work tests this. It also pre-empts the critique that "PSWE is just θ/δ power." | — |
| C4 | **Measured leakage inflation**: identical pipeline, window-level vs subject-level splits | Turns R1's claim into a number | R1, F8 |
| C5 | A **fully Kaggle-reproducible**, open pipeline: code, seeds, frozen folds, environment | — | R1 reproducibility critique |

**Research questions:** RQ1–RQ5 stay as in the README. Two are added:
- **RQ6:** Does the PSWE burden differ between AD, FTD and CN, after adjusting for age and sex? Does it correlate with MMSE within patients? MMSE is used for analysis only and is never a model input.
- **RQ7:** Does adding PSWE node features or a PSWE edge type to the best representation improve macro-F1, with significance tested on paired runs?

---

## 2. Compute: Kaggle only, with near-zero GPU

**The local machine is used only for writing code and git.** No data is stored locally and nothing runs there. Kaggle limits as currently reported: about 30 GPU-h/week (check your account page), 12 h per session, 4 CPU cores and about 16 GB RAM, and 20 GB of persistent `/kaggle/working`.

| Kaggle notebook | Accelerator | Inputs → Outputs | Est. time |
|---|---|---|---|
| NB00 `setup-tests` | CPU | code → pytest report, synthetic smoke run | 5 min |
| NB01 `download-preprocess` | CPU, Internet ON | OpenNeuro S3 (2.83 GB raw) → `windows/` (float16, ~350 MB), ICA logs, PSWE event tables | 1–1.5 h |
| NB02 `features-graphs` | CPU | windows → P1–P7 `.npz`, wPLI (5 bands), spatial/hybrid/PSWE graphs | ~1.5 h |
| NB03 `folds-baselines-pswe` | CPU | features → frozen fold JSON, demographics baseline, LR/SVM/RF/MLP, PSWE statistics | <1 h |
| NB04a–c `grid-shard-{1,2,3}` | **CPU** | 525 main runs, resumable `runs.csv` | ~3 h each, run in parallel |
| NB05a–c `ablations-shard-*` | CPU | ablations, permutation test, 10-repeat confirmation | ~3 h each |
| NB06 `deep-baselines` | **GPU T4: the only GPU notebook** | EEGNet + 1D-CNN, 25 runs each | **≤ 3 GPU-h** |
| NB07 `stats-figures` | CPU | all `runs.csv` → figures (PDF+PNG), LaTeX tables | 30 min |
| NB08 `external-ds004584` | CPU, Internet ON | PD cohort: ranking transfer + PSWE replication | ~3 h |

**Total GPU use: ≤ 3 h, about 10% of one week's quota. Everything else runs on CPU.** The model is about 3k parameters on 19-node graphs, so CPU training takes about 15–20 s per run.

Guardrails in code:
- **`bench` runs first.** It times one cell and refuses a shard whose projected time exceeds its `--max-hours`.
- **Resumable runs.** Each row of `runs.csv` is keyed by (pipeline, edge, repeat, fold, config_hash). A restarted session skips finished keys.
- **Batch execution.** Notebooks run via "Save Version → Save & Run All", which runs in the background up to 12 h and persists outputs. Each notebook's output is attached as the input of the next (Add Input → Notebook Output), so preprocessing is never recomputed.
- **Fixed hyperparameters, no tuning search.** The inner split is used for early stopping only.
- **Logged with every run:** `pip freeze`, git SHA, Kaggle image version, and the seed; deterministic torch flags are set.

**Getting the code onto Kaggle:** either push the branch so the notebooks can `pip install git+…@feat/implementation`, or zip `src/` into a private Kaggle Dataset. You choose at P0. I will not push without your go-ahead.

---

## 3. Repository additions

```
pyproject.toml, requirements-kaggle.txt (pinned), .gitignore
configs/default.yaml                 # every frozen hyperparameter: one source of truth
src/eegrep/
  data/download.py                   # HTTPS S3 lister (no AWS CLI), resumable, size-checked, raw only
  data/participants.py               # MMSE dropped from all model-facing tables (asserted)
  splits.py                          # repeated stratified subject-level folds + assert_disjoint
  preprocess.py                      # fixed preamble (MNE + ICLabel), joblib over subjects
  pswe.py                            # MPF sliding-window detector → events table (subject, ch, onset, dur, MPF)
  connectivity.py                    # vectorised wPLI/PLV/coherence per window & band
  features/p1_band p2_conn p3_dwt p4_time p5_raw p6_fused p7_pswe
  graphs.py                          # spatial | functional(α-wPLI) | hybrid | pswe-cooccurrence | random | identity; kNN, sym, D̃^-½(A+I)D̃^-½
  models.py                          # DenseGCN (19→64→32→3), DenseGAT, depth param, MLP-on-nodes (no-graph control)
  train.py  metrics.py  runner.py    # one run / all metrics / grid + bench + ablate (resumable, budgeted)
  baselines.py                       # demographics, LR/SVM/RF/MLP per representation (flattened), EEGNet, 1D-CNN
  explain.py                         # integrated gradients → node/edge importance per class
  stats.py  figures.py  tables.py    # §5–§6 below
splits/folds_ds004504.json           # committed, frozen before any training
tests/                               # see Verification
kaggle/NB00…NB08 .ipynb              # thin wrappers calling `python -m eegrep …`
```

---

## 4. Method decisions (locked)

**Data and preprocessing.** ds004504 v1.0.8 raw `.set` (not the authors' derivatives). The fixed preamble:
1. band-pass 0.5–45 Hz
2. 50 Hz notch
3. average reference
4. ICA with ICLabel, removing eye and muscle components with probability ≥ 0.8; fit per recording on a 1 Hz high-passed copy
5. resample to 128 Hz
6. 10 s windows (≈7,059)

ICA is per recording and unsupervised, so it cannot leak across subjects. Cross-subject scalers are fit on training subjects only. T3/T4/T5/T6 are renamed T7/T8/P7/P8 to get coordinates.

**Factor A: representations (7).**

| ID | Representation | Features per node |
|---|---|---|
| P1 | Welch relative power in 5 bands, spectral entropy, peak frequency | 7 |
| P2 | wPLI strength in each of the 5 bands | 5 |
| P3 | db4 DWT sub-bands D2–D5 and A5 × energy, entropy, mean \|c\|, std | 20 |
| P4 | mean, var, RMS, skew, kurtosis, ZCR, Hjorth mobility, Hjorth complexity, sample entropy, line length (line length replaces Hjorth activity, which equals var; deviation documented) | 10 |
| P5 | z-scored raw window | 1280 (parameter-count exception stated) |
| P6 | P1 + P3 + P2 + α clustering + α eigenvector centrality | 34 |
| **P7 (new, PSWE)** | fraction of window inside a PSWE, mean MPF, min MPF, local PSWE rate within ±60 s | 4 |

**Factor B: edges (3).** Spatial (10–20 coordinates, Gaussian kernel with σ = median pairwise distance), functional (α-band wPLI per window), and hybrid (their element-wise product). kNN with k = 4, symmetrised and normalised. **A PSWE co-occurrence graph** (Jaccard overlap of PSWE time between channel pairs, per recording) is tested in RQ7, not as a fourth factorial level, so the factorial stays clean.

**PSWE detector.**
- MPF is computed per channel in 2 s sliding windows. A PSWE is MPF < 6 Hz sustained for ≥ 5 s.
- The exact parameters (step size, the cortical-involvement or channel-count rule) are taken from the Methods of Milikovsky 2019 and the open-access PD paper (`10.3390/s23020918`) before coding, and recorded in the config.
- A sensitivity analysis covers the threshold (5/6/7 Hz) and the minimum duration (3/5/8 s).
- Detection runs per recording and uses no labels, so it cannot leak.

**Model and training: frozen.**
- Architecture: 2-layer GCN, 19→64→32→3, BatchNorm, dropout 0.5, mean pooling.
- Optimiser: AdamW, lr 1e-3, weight decay 1e-4, batch 64, at most 200 epochs, early stopping with patience 20 on validation macro-F1 computed at subject level.
- Loss weights: class weight / n_windows(subject), so every subject counts equally despite recording lengths of 307–1291 s.
- Subject prediction: the mean of the window logits.

**Seeds and CV: multiple seeds.**
- **5 repeats (seeds 0–4) × stratified subject-level 5-fold CV.** The seed sets both the partition and the weight initialisation. Inner 20% subject split for early stopping.
- The 3 best cells are re-run with **10 repeats** (seeds 0–9) to tighten confidence intervals.
- The main grid is 21 cells × 25 = **525 runs**. Every cell uses identical folds, so runs are paired.

---

## 5. Experiments, each with its justification

**Baselines (Table 4)**

| Baseline | Why it is there |
|---|---|
| Demographics only (age + sex, LR) | Confound check, **run first**. If it is far above chance (macro-F1 around 0.33), stop and adjust. |
| LR / SVM / RF / MLP on flattened features of each representation | A non-graph twin for every representation, which isolates the graph's contribution |
| EEGNet and 1D-CNN on raw windows (the only GPU use) | Standard deep EEG baselines (F14) |
| Published ds004504 results (F2, F3, Radwan 2025) | Context only; the caption flags that their protocols differ |

**Ablations (Table 5 and a forest plot).** Each runs on the best 1–2 cells with 5×5 CV.

| # | Ablation | Hypothesis / why a reviewer needs it |
|---|---|---|
| A1 | **No-graph control**: identity adjacency (MLP on nodes) | Does message passing help at all? Without this, the "GNN" claim is unsupported. |
| A2 | **Degree-matched random graph** | Does the specific topology matter, or only the act of smoothing? |
| A3 | k ∈ {3, 4, 6, fully connected} | The conclusions should not be an artefact of choosing k = 4. |
| A4 | Weighted vs binarised edges | F9 recommends weighted edges; this verifies it. |
| A5 | GCN depth ∈ {1, 2, 3} | Over-smoothing check; justifies using 2 layers. |
| A6 | GAT in place of GCN | Architecture sensitivity: is the representation ranking preserved? Measured with Kendall τ. |
| A7 | Window length ∈ {5, 10, 20} s | The literature uses 2–30 s; PSWEs need at least 5 s. |
| A8 | ICA on vs off | Sensitivity to the preprocessing preamble |
| A9 | Subject-equal loss weighting vs plain | Effect of the unequal recording lengths |
| A10 | **Leakage**: window-level vs subject-level split | Measures C4 / RQ5, likely the most-cited result |
| A11 | PSWE threshold and duration grid | Robustness of C2 and C3 |
| A12 | Demographic-residualised features | Is the EEG signal independent of age and sex? |

**PSWE arm (RQ6 and RQ7)**
1. Per-subject PSWE rate (events/min), the time fraction inside PSWEs, and topography.
2. Group comparison: Kruskal–Wallis with Dunn post-hoc tests (Holm-corrected), plus a negative-binomial GLM with age and sex covariates. Effect sizes (ε², rank-biserial r).
3. Within patients: Spearman correlation of PSWE rate with MMSE.
4. A partial correlation of PSWE rate with relative θ+δ power, which directly answers "is PSWE just slowing?"
5. RQ7 incremental value: best representation vs best + P7, and the best edge vs the PSWE co-occurrence edge, tested on paired 5×5 runs.

**External validation (RQ5 and PSWE replication).** ds004584: 100 PD, 49 controls, 63 channels. Its size is checked and you approve it before download. The test is whether the representation *ranking* transfers (Spearman ρ between the ranking orders), and whether PSWEs are uncommon in PD, which would replicate the 2023 Sensors finding.

**Permutation test.** The best cell is re-run 200 times with shuffled labels (1 repeat each, on CPU) to get an empirical p-value against chance.

---

## 6. Statistical analysis
- Per-cell mean ± SD, with **95% CIs** from a subject-level bootstrap (2,000 resamples) and a t-interval across the 25 runs.
- **Two-way repeated-measures ANOVA** (Representation × Edge) on macro-F1 with Greenhouse–Geisser correction and partial η². The interaction term answers RQ3.
- **Friedman test with Nemenyi post-hoc** across all 21 cells, shown as a **critical-difference diagram** (Demšar 2006).
- Pairwise comparisons: Wilcoxon signed-rank with Holm correction, and the **Nadeau–Bengio corrected repeated-CV t-test**, because fold scores share training data and are not independent.
- Effect sizes everywhere: Cohen's d (paired) and rank-biserial r.
- Classification diagnostics:
  - Cohen's κ
  - balanced accuracy
  - per-class sensitivity and specificity
  - ROC-AUC (one-vs-rest macro) and PR-AUC
  - **calibration** (Brier score, ECE)
- PSWE statistics as in §5. Every p-value family is Holm-corrected.

## 7. Figures and tables (journal set plus supplementary)

**Figures**

| # | Figure | Chart type |
|---|---|---|
| 1 | Pipeline: fixed preamble → 7 representations × 3 graphs → fixed GCN, plus the PSWE branch | Schematic |
| 2 | Cohort: age, sex and recording length per group | Violin and bar |
| 3 | Group PSDs (mean ± CI) and band-power **topomaps** for AD, FTD and CN | Line and topomap |
| 4 | **PSWE example**: EEG trace with the MPF time series and detected events shaded | Trace |
| 5 | **PSWE burden**: raincloud per group, PSWE-rate topomaps, PSWE rate vs MMSE scatter with regression | Raincloud, topomap, scatter |
| 6 | Main effect of representation | Box and strip with CI |
| 7 | Main effect of edge type | Box and strip with CI |
| 8 | **7×3 interaction**: heatmap plus an interaction line plot | Heatmap and line |
| 9 | **Critical-difference diagram** | CD diagram |
| 10 | **Confusion matrices**: normalised 3×3 for the best cell of each representation, aggregated over 25 runs | Matrix grid |
| 11 | ROC and PR curves, one-vs-rest per class, mean ± band | ROC / PR |
| 12 | **Leakage**: subject-level vs window-level | Paired slope and bar |
| 13 | Ablations: Δ macro-F1 with 95% CI | **Forest plot** |
| 14 | Mean connectivity graphs per group, plus AD−CN and FTD−CN difference graphs | **Circular connectome** plots |
| 15 | Explainability: integrated-gradients node importance per class | Topomaps and edge-importance plots |
| 16 | Learned subject embeddings coloured by class | t-SNE / UMAP |
| 17 | External validation: representation ranking ds004504 → ds004584 | Slope chart |
| S1–S5 | Training curves, calibration reliability diagrams, seed-variance strip plot, PSWE threshold-sensitivity heatmap, permutation-null histogram | Supplementary |

**Tables**

| # | Table |
|---|---|
| 1 | Positioning vs prior work, with Radwan 2025 added |
| 2 | Demographics |
| 3 | Full metrics per cell |
| 4 | Baselines and published results |
| 5 | Ablations |
| 6 | ANOVA, Friedman and corrected-t results |
| 7 | PSWE statistics |
| S1 | Environment and versions |

Output: vector PDF plus 300-dpi PNG, colour-blind-safe palette, and LaTeX tables generated straight from `runs.csv`. No hand-typed numbers.

---

## 8. Phases and gates (implementation starts only when you say go)

**Step 0, after you approve:** save this plan as `IMPLEMENTATION_PLAN.md` in the repo and commit it on a new **local** branch, `feat/implementation`. No code runs and nothing is downloaded.

| Phase | Week | Work | Gate before moving on |
|---|---|---|---|
| P0 | 0 | Scaffold, config, tests, NB00. Verify the new DOIs (Milikovsky 2019, Senatorov 2019, s23020918, EMBC 2025, Radwan 2025, Nation 2019 BBB) against Crossref, as the repo's standard requires | Tests pass on Kaggle; DOIs logged to `doi_tracking.csv` |
| P1 | 1 | NB01: download, preprocessing, PSWE detection | ≈7,059 windows; durations 307–1291 s; ICA log; PSWE detector reproduces the published criteria on a synthetic slow-wave test |
| P2 | 2 | NB02: P1–P7 and all graphs | Shapes, NaN and symmetry checks; AD shows a higher θ/α ratio than CN (sanity check) |
| P3 | 3 | NB03: frozen folds, **demographics baseline**, classical baselines, PSWE statistics | Disjointness assertion; demographics macro-F1 near chance; any result above 95% triggers a bug hunt |
| P4 | 4–5 | NB04 shards: 525-run grid; NB06: EEGNet (GPU ≤ 3 h) | 525 rows; no config-hash drift; GPU-hours logged |
| P5 | 6 | NB05: ablations A1–A12, RQ7, 10-repeat confirmation, permutation test | All rows present |
| P6 | 7 | NB08: ds004584 (after your approval) | Ranking-transfer ρ reported |
| P7 | 8 | NB07: statistics, figures and tables | Every number traces back to a `runs.csv` row |
| P8 | 9–10 | Update `paper_structure.md` (new RQ6/RQ7, Figs 4–5, Table 7, BBB related-work subsection), `README.md`, `final_decision.md`; draft the manuscript | Submission checklist in `paper_structure.md` |

There is one local commit per phase on `feat/implementation`. Nothing is pushed without your approval.

## 9. Risks and honest framing
- **"PSWE is just slowing."** Addressed by the partial correlation and the RQ7 incremental test. A null result is still reported.
- **No BBB ground truth in ds004504.** The paper always says "BBB-associated marker", cites the evidence chain, and lists this as a limitation; DCE-MRI validation goes in Future Work.
- **88 subjects.** Addressed by repeated CV, a conservative corrected t-test, bootstrap CIs, a permutation test and external validation.
- **Kaggle limits.** Runs are sharded and resumable, and the work is almost entirely CPU.

## Verification
- `pytest` runs in NB00 on Kaggle.
  - Subject disjointness across train, validation and test for every repeat and fold.
  - MMSE is absent from every model input.
  - Scalers are fit on training subjects only.
  - wPLI on synthetic signals: lagged signals give ≈ 1, zero-lag signals give ≈ 0.
  - PSWE detector on synthetic signals: an injected 4 Hz burst lasting 6 s is detected; a 4 s burst is not.
  - Graphs are symmetric, with spectral radius ≤ 1.
  - Feature shapes are 7/5/20/10/1280/34/4.
  - The GCN has about 3k parameters.
  - A runner resume test: rerunning skips finished keys.
- Synthetic end-to-end smoke test (folds → features → bench) in under 2 min on CPU, in NB00.
- Real-data gates from §8.
- Compute check: the sum of the `runs.csv` timing column per accelerator must be ≤ 3 GPU-h.

---

## Implementation log

### P0 (scaffold, tests, NB00, DOI verification)

**Implemented:** package `src/eegrep` with these modules:
- `config`, `data.download`, `data.participants`, `splits`
- `connectivity`, `graphs`, `pswe`, `features`
- `models`, `metrics`, `train`, `cache`, `runner`
- `synthetic`, `cli`

Also `configs/default.yaml`, 7 test modules, and the `kaggle/nb00-setup-tests` kernel.

**DOIs verified on Crossref (2026-10-09)** and appended to `doi_tracking.csv` as B1–B7:
- Milikovsky 2019 (STM)
- Senatorov 2019 (STM)
- Milikovsky 2023 (Sensors)
- Ganon 2025 (EMBC)
- Nation 2019 (Nat Med)
- Radwan 2025 (bioRxiv)
- Nadeau & Bengio 2003

Demšar 2006 (JMLR) has no DOI, so it will be cited by URL.

**Dataset version.** The S3 bucket serves v1.0.9. Its `CHANGES` entry is "Added in 'How to cite' section", so the EEG data is identical to v1.0.8. We keep citing the v1.0.8 DOI, and the downloader records the served version plus a SHA-256 for every file.

**PSWE parameters**, taken from 10.3390/s23020918 (open access):
- FFT on 2 s windows with a 1 s overlap
- an event is "MPF lower than 6 Hz for five consecutive seconds or more"
- the MPF frequency range is not reported, so we use 1–45 Hz (our pass-band), with sensitivity tested in A11
- the source paper averaged channels into 9 regions on a 60-channel cap; we detect per channel (19 channels), because P7 node features need channel resolution

**Deviations from §3 and the Verification section:**
- Features live in a single `features.py` instead of 7 files.
- Run results are written to `runs.jsonl`, one JSON object per run, instead of `runs.csv`. The nested per-subject predictions don't fit a CSV, and JSONL appends safely.
- The PSWE synthetic test uses a 10 s burst (must be detected) and a 2 s burst (must not be). With 2 s windows at a 1 s step, a 4 s burst can produce 5 sub-threshold samples, which would make the test ambiguous. The rule itself is unit-tested exactly on constructed MPF sequences.
- Preprocessed recordings enter the cache in µV.

**P0 gate: passed on 2026-10-09** (Kaggle NB00 v3, git `169fa6f`, Python 3.13.15, torch 2.11 CPU, mne 1.13.2).
- 49/49 tests pass.
- The synthetic end-to-end smoke test (12 cells plus the resume check) finished in 31 s on CPU.
- Reports are in `reports/`: `nb00_pytest.txt`, `nb00_environment.json` and `kaggle_pip_freeze.txt`, the resolved environment lock.

**Early P3 items, run in parallel at the user's request** (Kaggle NB03a):
- The frozen folds are committed as `splits/folds_ds004504.json`: 88 subjects, 5×5, test folds of 17–18 subjects.
- **Demographics confound check:** an age + sex logistic regression scores macro-F1 **0.411 [0.364, 0.458]**, against **0.302** for stratified chance. This is moderately above chance, consistent with the sex imbalance (AD 67% F vs CN 38% F). It is not a blocker, but:
  - every EEG result must be compared with 0.41, not 0.33;
  - ablation A12 (demographic-residualised features) is mandatory.

### P1/P2 gates and the RQ6 result (2026-10-09, Kaggle NB01×3, NB02, NB03b)
- **NB02 QC passed with no gate failures:**
  - 88 subjects (36/23/29) and 7,013 windows;
  - durations of 307.1–1291.1 s, matching the dataset files exactly;
  - ICA removed 2.4 components per subject on average (157 eye, 51 muscle);
  - θ/α ratio is AD 1.57 > FTD 0.68 > CN 0.28, p = 1.3e-7.
- **RQ6 (PSWE), detailed in `reports/rq6/`:**
  - After adjusting for age and sex, a negative-binomial regression gives PSWE rate ratios vs CN of **AD 2.96 [1.58–5.57], p = 0.0007** and **FTD 2.07 [1.03–4.15], p = 0.041**. This replicates Milikovsky 2019 for AD in an independent open cohort, and is the first such report for FTD.
  - **However,** the PSWE rate correlates ρ = 0.88 with relative δ+θ power. After additionally adjusting for that slowing, the AD rate ratio becomes 0.50 (p = 0.0009).
  - **Interpretation:** the fixed-threshold detector (MPF < 6 Hz) mostly tracks background slowing, and AD slowing is continuous rather than paroxysmal.
  - PSWE vs MMSE within patients: ρ = −0.22, p = 0.095.
  - Planned consequences:
    1. report this as the pre-registered "is PSWE just slowing?" result;
    2. expect RQ7 to be near-null;
    3. treat a background-relative PSWE definition as an *exploratory* follow-up, labelled post hoc. This one awaits the user's decision.
