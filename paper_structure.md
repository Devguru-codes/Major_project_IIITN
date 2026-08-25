# Paper Structure — Detailed Blueprint

**Working title (preferred):**
> **Representation Matters: A Controlled Comparison of EEG Signal Representations and Graph Constructions for Dementia Classification with a Fixed Graph Neural Network**

**Alternative titles**
- *Does the Representation or the Model Matter? Disentangling Signal Encoding from Graph Topology in EEG-Based Dementia Classification*
- *A Factorial Benchmark of EEG Representations and Brain-Graph Constructions for AD/FTD Classification under a Fixed GCN*
- *Isolating Signal Representation from Model Architecture in Graph-Based EEG Dementia Diagnosis*

**Paper type:** Methodological / controlled benchmark study (not an architecture-novelty paper — this is deliberate and must be signalled from the title onward).

**Target length:** 7,500–9,000 words · 6 figures · 5 tables · 45–60 references

---

## Target venues (ranked)

| Rank | Venue | Why | Fit risk |
|---|---|---|---|
| 1 | **Journal of Neural Engineering** (IOP) | Rewards methodological rigour and negative/controlled results; already hosts the key EEG-DL reviews and EEGNet | Wants clear neuro-engineering relevance — lead with the clinical framing |
| 2 | **IEEE J. Biomedical & Health Informatics** | Strong on benchmark + reproducibility papers | May push for architectural novelty |
| 3 | **Computers in Biology and Medicine** (Elsevier) | Published R16; comfortable with pipeline comparisons | Slightly lower methodological bar |
| 4 | **IEEE TNSRE** | Home of F3/F4/F5 — the exact conversation we're joining | Reviewers will be the GNN authors; be rigorous |
| 5 | **Cognitive Neurodynamics** (Springer) | Published R1 and R14; receptive to this dataset | Lower impact |
| 6 | **Scientific Reports** | Safe fallback, broad scope | Least prestige for a methods paper |

**Recommendation:** aim at **Journal of Neural Engineering** first. Its readership already accepts that evaluation protocol is a scientific result, which is central to our contribution.

---

## Abstract (structured, 250 words)

Write this **last**. Fill the bracketed numbers only after results are final.

| Move | Sentences | Content |
|---|---|---|
| **Objective** | 2 | EEG dementia classification reports 80–99% accuracy on the same public dataset, but studies vary preprocessing *and* model simultaneously, so the contribution of the signal representation is unknown. We isolate it. |
| **Approach** | 3–4 | 88 subjects (36 AD / 23 FTD / 29 CN), ds004504. Six representations × three graph constructions, fully crossed, under a **single fixed 2-layer GCN**. Subject-level nested 5-fold CV × 5 seeds (450 runs). |
| **Main results** | 3–4 | Best representation = [X], macro-F1 [X.XX ± X.XX]. Effect of representation [ηp² = X]; effect of edge definition [ηp² = X]; interaction [significant / not]. Window-level splitting inflates macro-F1 by [X] points on an identical pipeline. |
| **Significance** | 2 | Representation and protocol effects are quantified and separable from architecture. Provides a calibration factor for interpreting existing results on this benchmark. Code, seeds and fold assignments released. |

**Keywords:** EEG; graph neural network; Alzheimer's disease; frontotemporal dementia; signal representation; functional connectivity; reproducibility; data leakage

---

# 1. Introduction (~1,100 words, 4 paragraphs + contributions)

**Job of this section:** move from clinical need → methodological confusion → our precise question. Do *not* claim a new architecture.

### ¶1 — Clinical motivation (~180 words)
Dementia burden; AD vs FTD differential diagnosis is genuinely difficult and clinically consequential; EEG is cheap, non-invasive and widely available versus PET/MRI.
**Cite:** F12 (AD as a network disorder), F9 (AD and FTD produce *dissimilar* network disturbance — this sets up the 3-class task), R15, R3.

### ¶2 — What the field has done (~250 words)
Progression: spectral markers → handcrafted features + classical ML → CNNs → transformers → GNNs. State that reported accuracy on ds004504 spans roughly 80–99%.
**Cite:** F10, F2 (DICE-Net, 83.28% AD-CN under LOSO), R14, R15, F5 (GNN-EEG survey), F3, F4, F6, R9, R10, R12.

### ¶3 — The problem (~300 words) ← **the pivotal paragraph**
Three confounds are tangled together in the literature: (i) signal representation, (ii) graph construction, (iii) model architecture — plus (iv) evaluation protocol. Deploy the independent evidence here rather than asserting it yourself:
- **R1** (2026 scoping review of this dataset): gains *"arise not from inherent representational advantage, but from data leakage and overfitting to non-independent samples"*; added representations *"do not demonstrate a systematic performance improvement."*
- **R2** (2026): compared five time-frequency representations under one CNN — but *"does not explicitly model the true spatial topology… alternative graph-based or topology-aware fusion schemes may further enhance… performance."*
- **R9** (2025): *"most studies overlook spatial connectivity between channels."*
- **F6**: CNNs learn on Euclidean space, *"neglecting the functional connectivity features."*
- **F8**: leakage as a general crisis in ML-based science.

### ¶4 — What we do (~200 words)
State the design in one breath: 6 representations × 3 edge definitions, one fixed GCN, subject-level splits, 450 runs, one dataset, one protocol. Note the deliberate choice of a *simple* model — the model is the control, not the contribution.
**Cite:** F13 (GCN), F11 (PLI/wPLI), R8 and F7 as the explicit calls for this work.

### ¶5 — Contributions (bulleted, 4 items)
1. **Methodological** — a factorial protocol separating representation, topology and architecture.
2. **Empirical** — the first systematic evidence on which representation a GNN benefits from for AD/FTD/CN, with effect sizes and CIs.
3. **Scientific** — evidence on whether AD/FTD are better separated by node-local spectral properties or inter-channel coupling.
4. **Reproducibility** — released code, seeds, and frozen fold assignments on a CC0 dataset.

> **Writing note:** end ¶3 with the gap sentence and open ¶4 with "We therefore…". Reviewers look for exactly that hinge.

---

# 2. Related Work (~1,200 words, 4 subsections)

Organise by **what varies**, not chronologically. This framing is itself an argument.

### 2.1 Signal representations for EEG dementia classification (~350 w)
Sub-groups: spectral/band power (R3, F2, R15); time-domain/complexity (R15, R14); time-frequency (R2, R16); image-like encodings (R2, R6, R10).
**Key point:** R2 and R6 show that representation choice is treated as a first-class question in 2026 — but always with CNNs.

### 2.2 Graph neural networks for EEG (~350 w)
Foundations (F13 GCN, GAT, ChebNet, GraphSAGE, GIN) → EEG-specific (F5 survey) → disease-specific (F3, F4, F6, R9, R12, F7).
**Key point:** F3 is the closest precedent — it compares functional-connectivity methods as GNN inputs but holds node features fixed. Say this explicitly and generously; it is the paper our reviewers most likely wrote.

### 2.3 Graph construction for EEG (~250 w)
Spatial vs functional vs learned vs hybrid. Connectivity measures and volume conduction (F11, wPLI/Vinck 2011, F9, F12, R13, R16).
**Key point:** R9's admission that spatial connectivity is usually overlooked motivates Factor B.

### 2.4 Evaluation protocol and leakage (~250 w)
F8, R1, plus the window-level vs subject-level split problem specific to EEG.
**Key point:** this subsection earns the right to run the leakage ablation in §4.

### 2.5 Positioning table — **Table 1**
| Study | Varies representation? | Varies graph? | Fixed model? | Subject-level split? | Public data? |
Rows: F2, F3, F4, F6, R2, R9, R10, R12, R14, **Ours**.
Only our row has ✓ in all five columns. This single table is the strongest argument in the paper — build it early and let it drive the narrative.

---

# 3. Materials and Methods (~2,800 words — the heart of the paper)

### 3.1 Dataset (~350 w)
ds004504 (AHEPA / Ioannina). Report the values verified directly from the dataset files, and say that you verified them there:
88 subjects (36 AD / 23 FTD / 29 CN) · 19 channels 10–20 · 500 Hz · eyes-closed rest · A1–A2 reference · Nihon Kohden EEG 2100 · 0.4–50 Hz applied · **19.61 h** total · per-subject 307–1291 s (mean 802 s) · **CC0**.
Demographics table (**Table 2**): age, sex, MMSE per group.
**Cite:** F1, dataset DOI `10.18112/openneuro.ds004504.v1.0.8`.
> ⚠ Footnote the broken v1.0.9 DOI in the dataset's own metadata — a small, concrete reproducibility observation that reviewers appreciate.

### 3.2 Preprocessing — the fixed preamble (~350 w)
One identical code path for every condition: band-pass 0.5–45 Hz → 50 Hz notch → average reference → ICA with ICLabel (drop eye/muscle ≥ 0.8) → resample 128 Hz → 10 s non-overlapping windows → ≈7,059 windows.
Justify 10 s via R10. State that all fitted transforms are training-fold-only.
**Cite:** EEGLAB, MNE, ICLabel, autoreject, PREP, DISCOVER-EEG, R7.

### 3.3 Factor A — six representations (~700 w)
One short subsection each (P1–P6) with the exact output shape. Include a small equation block per pipeline. **This is where reviewers check competence — give exact dimensions and parameter values (Welch window, wavelet family and levels, band edges).**
**Cite per pipeline:** P1 → R3, F2; P2 → F11, R13; P3 → R2, R16; P4 → R15; P5 → F14; P6 → R12.

### 3.4 Factor B — three graph constructions (~400 w)
Spatial (10–20 coordinates, Gaussian kernel) · functional (wPLI) · hybrid (element-wise product). kNN k=4, symmetrised `A = max(A, Aᵀ)`, weighted, undirected, normalised `Â = D̃^(-1/2)(A+I)D̃^(-1/2)`.
**Cite:** F5, F11, R9, F9 (weighted graphs).

### 3.5 Fixed GNN (~350 w)
2-layer GCN, 19→64→32→3, mean-pool, ~3k params. State the full justification: it is the *control*, it matches the data scale, 2 hops suffice on a 19-node graph, deeper risks over-smoothing.
Explicitly say why GAT/ChebNet/GraphSAGE/GIN/ST-GCN were **not** chosen as primary — pre-empts the top reviewer objection.
**Cite:** F13 primary; GAT/ChebNet/GraphSAGE/GIN for the rejection rationale; F7 for the call to compare architectures.

### 3.6 Experimental design and evaluation (~450 w)
6×3 factorial = 18 cells × 5 folds × 5 seeds = 450 runs. Subject-level stratified nested 5-fold CV; frozen fold assignments. Subject-level prediction by mean of window logits. Controlled variables list. Hyperparameters fixed a priori and identical everywhere (AdamW, lr 1e-3, wd 1e-4, batch 64, ≤200 epochs, early stopping patience 20 on val macro-F1, inverse-frequency class weights).
**Metrics:** macro-F1 primary; balanced accuracy, ROC-AUC (OvR macro), PR-AUC, per-class sensitivity/specificity, Cohen's κ, confusion matrix. State plainly that accuracy is reported but never used for ranking.
**Cite:** F8, R1.

### 3.7 Statistical analysis (~250 w)
Two-way repeated-measures ANOVA (Representation × Edge), interaction term of primary interest; Wilcoxon signed-rank on paired folds with Holm–Bonferroni; effect sizes (partial η², Cohen's d); 95% CIs.

### 3.8 Baselines (~200 w)
Demographics-only (age+sex) — the confound check; LR / SVM / RF on P1; EEGNet (F14); 1D-CNN; published GNN results (F3, F4) with an explicit note on protocol incomparability.

### 3.9 Ablations (~200 w)
(1) k ∈ {3,4,6}; (2) window ∈ {5,10,20} s; (3) weighted vs binarised edges; (4) GCN depth ∈ {1,2,3}; (5) GAT substitution; (6) **leakage ablation** — window-level vs subject-level splitting on an identical pipeline.

---

# 4. Results (~1,800 words — report, do not interpret)

### 4.1 Confound check (~150 w) — **report first**
Demographics-only baseline. If it is near chance, you have earned the right to interpret everything that follows. Put it first for exactly that reason.

### 4.2 Main effect of representation (~400 w) — **Figure 2**, **Table 3**
Macro-F1 per pipeline (mean ± 95% CI), pooled over edge types. Full metric table.

### 4.3 Main effect of graph construction (~300 w) — **Figure 3**
Spatial vs functional vs hybrid, pooled over representations.

### 4.4 Interaction (~350 w) — **Figure 4** (6×3 heatmap)
The novel result. Report the ANOVA interaction term and whether the best representation depends on edge definition.

### 4.5 Comparison with baselines and prior work (~300 w) — **Table 4**
Our numbers vs classical ML, EEGNet, and published results — with the incomparability caveat stated in the caption, not buried.

### 4.6 Ablations (~200 w) — **Table 5**
### 4.7 Leakage ablation (~250 w) — **Figure 5** ← likely the most-cited item
Identical pipeline, identical model, identical data; subject-level vs window-level splitting side by side. Report the inflation in points.

### 4.8 External validation (~200 w) — **Figure 6**
ds004584 (100 PD / 49 control, 63 ch). Test whether the *ranking* transfers, not the absolute values. Say so in the caption.

> **Discipline:** Results contains no "because", "suggests" or "indicates". All interpretation lives in §5. Reviewers notice.

---

# 5. Discussion (~1,600 words)

### 5.1 Principal findings (~300 w)
Three or four sentences of plain answers to RQ1–RQ5, then unpack.

### 5.2 Does representation or topology matter more? (~350 w)
Interpret the effect sizes. Relate to R1's convergence finding: if effects are small, that *corroborates* an independent 2026 review rather than being a failure.

### 5.3 Neurophysiological interpretation (~350 w)
If connectivity representations win → supports the network-degeneration account (F12, F9, R3, R13). If spectral node features win → supports local oscillatory slowing as the dominant marker (R15, R3). Discuss AD vs FTD confusion patterns from the confusion matrix against F9's dissimilar-network finding.

### 5.4 Implications for how this field reports results (~300 w)
Use the leakage ablation to offer a calibration factor for reading existing ds004504 numbers. Anchor to R1 and F8. **Be constructive, never accusatory — name no paper as guilty.**

### 5.5 Limitations (~300 w) — write this honestly and early
88 subjects; ~18-subject test folds; single site/device/protocol; FTD is the minority class (n=23); unequal recording lengths; electrode-level rather than source-level graphs (F7's point); eyes-closed only; no longitudinal or external clinical validation.

---

# 6. Conclusion (~250 words)
No new information. Restate the question, the design, the headline numbers, and the one sentence you want cited. Close on reproducibility.

---

# 7. Future Work (~500 words, 6 items)

| # | Direction | Grounding |
|---|---|---|
| 1 | **Source-space graphs** — repeat with source-localised nodes instead of electrodes | F7 explicitly requests this |
| 2 | **Learned adjacency** — let the graph be a parameter, compare against fixed edges | F4 (adaptive gated GCN) |
| 3 | **Temporal graphs** — ST-GCN / dynamic connectivity over window sequences | R10 (time-graphs), F6 |
| 4 | **Architecture sweep** — GAT, GIN, graph transformers under the *same* representation, i.e. transpose our design | F7, F5 |
| 5 | **Cross-dataset and cross-device generalisation** — BrainLat, CAUEEG, TDBRAIN; add an MCI class | R1 on electrode-topology adaptation |
| 6 | **Clinical extension** — MCI-to-dementia conversion, severity regression against MMSE | R1, R3 |

Add one honest sentence: the most valuable next study may be a **multi-site replication** of this exact protocol, since our single-site design cannot establish device invariance.

---

# 8. Reproducibility Statement (~150 words)
Dataset DOI (CC0), code repository, exact package versions, random seeds {0–4}, frozen fold-assignment JSON, hardware, total compute. State that the study is re-runnable end-to-end by a third party.

---

# Figure inventory

| # | Figure | Type | Purpose |
|---|---|---|---|
| 1 | Pipeline overview | Schematic | Raw EEG → fixed preamble → Factor A (6) × Factor B (3) → fixed GCN. **Must convey "one thing varies at a time" at a glance** |
| 2 | Macro-F1 by representation | Box/violin + CI | Main effect A |
| 3 | Macro-F1 by edge definition | Box/violin + CI | Main effect B |
| 4 | 6×3 performance heatmap | Heatmap | Interaction — the novel result |
| 5 | **Subject-level vs window-level split** | Paired bar | The leakage result |
| 6 | Ranking transfer to ds004584 | Slope/parallel-coords | Generalisation |
| S1–S3 | Confusion matrices, connectivity topographies, training curves | Supplementary | |

# Table inventory

| # | Table | Purpose |
|---|---|---|
| 1 | Positioning vs prior work (5 ✓/✗ columns) | The gap, made visual |
| 2 | Dataset demographics (age, sex, MMSE per group) | Transparency |
| 3 | Full metrics per cell | Primary results |
| 4 | Baselines and prior published results | Contextualisation |
| 5 | Ablation summary | Robustness |

---

# Citation map — where each of the 30 selected papers is used

| Ref | DOI | §1 | §2 | §3 | §5 | §7 | Primary role |
|---|---|---|---|---|---|---|---|
| **R1** | 10.1007/s11571-026-10464-w | ✓ | ✓ | ✓ | ✓ | ✓ | **Core gap evidence + leakage + SOTA anchor** |
| **R2** | 10.3390/diagnostics16050746 | ✓ | ✓ | ✓ | ✓ | | **Closest prior work; graph recommendation (Cat B)** |
| R3 | 10.1038/s41598-026-42452-9 | ✓ | ✓ | ✓ | ✓ | ✓ | Spectral + FC biomarkers |
| R4 | 10.1038/s41598-026-57069-1 | | ✓ | | ✓ | | Explainable ML comparison |
| R5 | 10.3390/brainsci16080856 | | ✓ | | ✓ | ✓ | Source connectivity + microstates |
| R6 | 10.3390/brainsci16070716 | | ✓ | ✓ | | | Representation encoding is a live question |
| R7 | 10.1016/j.mex.2026.103821 | | ✓ | ✓ | | | Preprocessing comparison precedent |
| **R8** | 10.1038/s41598-025-02018-7 | ✓ | ✓ | | ✓ | | **Category A — explicit GCN future work** |
| R9 | 10.3389/fnins.2025.1555657 | ✓ | ✓ | ✓ | ✓ | | Spatial-vs-functional edge gap |
| R10 | 10.3390/diagnostics15111441 | | ✓ | ✓ | ✓ | ✓ | Time-graphs; 10 s window justification |
| R11 | 10.1142/s0129065725500480 | | ✓ | | ✓ | | Graph spectral AD/FTD |
| R12 | 10.1007/s40747-025-01974-x | | ✓ | ✓ | ✓ | | Multi-connectivity GCN (one-factor ancestor) |
| R13 | 10.1186/s12938-025-01361-0 | | ✓ | ✓ | ✓ | | Phase synchronisation networks |
| R14 | 10.1007/s11571-025-10232-2 | ✓ | ✓ | | ✓ | | SOTA baseline |
| R15 | 10.1038/s41514-025-00243-y | ✓ | ✓ | ✓ | ✓ | | Complexity features (P4) |
| R16 | 10.1016/j.compbiomed.2025.111041 | | ✓ | ✓ | ✓ | | Morlet + MI connectivity (P3/P2) |
| **F1** | 10.3390/data8060095 | | | ✓ | | | **Dataset origin paper** |
| **F2** | 10.1109/access.2023.3294618 | ✓ | ✓ | ✓ | ✓ | | **Primary benchmark (83.28%, LOSO)** |
| **F3** | 10.1109/tnsre.2022.3204913 | ✓ | ✓ | ✓ | ✓ | | **Closest GNN precedent** |
| F4 | 10.1109/tnsre.2023.3321634 | | ✓ | ✓ | ✓ | ✓ | Learned adjacency comparison |
| **F5** | 10.1109/tnsre.2024.3355750 | ✓ | ✓ | ✓ | | ✓ | **GNN-EEG taxonomy** |
| F6 | 10.1002/hbm.25994 | ✓ | ✓ | | ✓ | ✓ | CNNs neglect connectivity |
| **F7** | 10.1038/s41746-023-00983-9 | ✓ | ✓ | ✓ | ✓ | ✓ | **Category A — architecture + source-space calls** |
| **F8** | 10.1016/j.patter.2023.100804 | ✓ | ✓ | ✓ | ✓ | | **Leakage framework** |
| **F9** | 10.1186/1471-2202-10-101 | ✓ | ✓ | ✓ | ✓ | | **AD ≠ FTD networks; weighted graphs (Cat B)** |
| F10 | 10.1016/j.neunet.2019.12.006 | ✓ | ✓ | | | | High-impact ML baseline |
| **F11** | 10.1002/hbm.20346 | | ✓ | ✓ | ✓ | | **PLI — edge definition** |
| F12 | 10.1093/cercor/bhj127 | ✓ | ✓ | | ✓ | | AD as a network disorder |
| **F13** | 10.48550/arXiv.1609.02907 | ✓ | ✓ | ✓ | | | **GCN — the fixed model** |
| F14 | 10.1088/1741-2552/aace8c | | ✓ | ✓ | ✓ | | EEGNet baseline |

### Supporting citations to add (~15–25, not in the 30)
Tools: EEGLAB `10.1016/j.jneumeth.2003.10.009` · MNE `10.1016/j.neuroimage.2013.10.027` · ICLabel `10.1016/j.neuroimage.2019.05.026` · autoreject `10.1016/j.neuroimage.2017.06.030` · PREP `10.3389/fninf.2015.00016` · DISCOVER-EEG `10.1038/s41597-023-02525-0`
Architectures: GAT `10.48550/arXiv.1710.10903` · ChebNet `10.48550/arXiv.1606.09375` · GraphSAGE `10.48550/arXiv.1706.02216` · GIN `10.48550/arXiv.1810.00826`
Connectivity: wPLI `10.1016/j.neuroimage.2011.01.055` · PLV `10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C`
Datasets: ds004584 `10.18112/openneuro.ds004584.v1.0.0` · BrainLat `10.1038/s41597-023-02806-8` · CAUEEG `10.1016/j.neuroimage.2023.120054` · TDBRAIN `10.1038/s41597-022-01409-z`
Reviews: `10.1088/1741-2552/ab0ab5` · `10.1088/1741-2552/ab260c` · `10.3390/bioengineering10030372`
Clinical epidemiology: add 2–3 current dementia-prevalence references for ¶1.

---

# Recommended writing order

Not front-to-back. Write in the order that locks facts down first:

1. **§3 Methods** — while implementing. If you can't write it, the design isn't settled.
2. **Table 1** (positioning) — this defines the paper's argument.
3. **Figure 1** (pipeline schematic) — forces design clarity.
4. **§4 Results** — as numbers arrive; captions first, prose second.
5. **§2 Related Work** — after Results, so emphasis matches findings.
6. **§5 Discussion** → **§7 Future Work** → **§6 Conclusion**.
7. **§1 Introduction** — second-to-last, once you know what you actually found.
8. **Abstract** — last.

---

# Pre-empting the four objections you will get

| Objection | Where to answer | Answer |
|---|---|---|
| *"Why such a simple model? Use a transformer/ST-GCN."* | §3.5 + §5.5 | The model is the **control variable**. Varying it would destroy the comparison. GAT is included as sensitivity; ST-GCN is future work. Say this in §3.5 — do not wait for review. |
| *"88 subjects is too small."* | §3.1, §4.1, §5.5 | Acknowledged openly. Mitigated by 5×5 repeated CV, paired statistics, effect sizes, CIs, and external validation. Never claim clinical readiness. |
| *"Accuracy is lower than published 99% results."* | §4.5 caption + §5.4 | **This is a finding, not a weakness.** Cite R1 and show the leakage ablation. Pre-empt it in the Results caption so no reviewer discovers it as a "problem". |
| *"Not novel — it's just benchmarking."* | §1 ¶4, Table 1 | Novelty is the **crossed factorial with a fixed model plus the quantified protocol effect**. Table 1 shows no prior work does all five. Cite R8 and F7 as explicit third-party requests for this study. |

---

# Submission checklist

- [ ] All 30 refs cited at least once; citation map above satisfied
- [ ] Every reported number traceable to a logged run
- [ ] Zero interpretation in §4; zero new results in §5
- [ ] Limitations names the 18-subject test fold explicitly
- [ ] Confound check (age+sex) reported **before** main results
- [ ] Leakage ablation included and discussed constructively
- [ ] Dataset cited as v1.0.8 (**not** the broken v1.0.9)
- [ ] MMSE confirmed absent from all model inputs
- [ ] Code, seeds, fold JSON public before submission
- [ ] CC0 dataset licence acknowledged
- [ ] Ethics: public de-identified secondary data — state it
- [ ] Any accuracy >95% re-investigated as a suspected bug
