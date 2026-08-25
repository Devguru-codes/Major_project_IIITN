# Deep Literature Review & Research Direction Report
## Neurological Disorder Classification Using Signal Processing + Graph Neural Networks

**Prepared:** 17 August 2026
**Discovery corpus:** 1,091 unique DOI-bearing works harvested from OpenAlex across 37 structured queries
**Examined in detail:** 87 papers
**Selected:** 30
**All citation counts checked:** 17 August 2026, against three independent sources (Crossref, OpenAlex, Semantic Scholar)

> **Verification standard used throughout.** Every DOI in this report was resolved against Crossref (76), DataCite (6, arXiv/OpenNeuro), or by HTTP resolution through doi.org. No DOI is written from memory. Where a paper is paywalled and its full text could not be retrieved, this is stated explicitly and no claim is made about its internal content. Where a field could not be verified, it is written **"Not reported"** or **"Not verified"** rather than inferred.

---

# Section 1 — Executive Summary

**Best disease:** Alzheimer's disease **and** frontotemporal dementia, treated jointly as a dementia-subtype problem (score 8.10/10; next best Parkinson's 7.65).

**Best dataset:** OpenNeuro **ds004504** — the AHEPA / University of Ioannina resting-state EEG corpus. Verified directly from the dataset files: **88 subjects (36 AD, 23 FTD, 29 CN)**, 19 channels (10–20), 500 Hz, eyes-closed, **19.61 hours** total, **CC0** licence.

**Best classification task:** **3-class subject-level AD vs FTD vs CN**, single acquisition protocol, no dataset merging. Binary AD vs CN reported additionally for comparability with prior work.

**Recommended preprocessing pipelines:** 6 pipelines spanning raw, time-domain, frequency-domain, time-frequency, connectivity, and fused representations (Section 9).

**Recommended GNN:** a **2-layer GCN** (Kipf & Welling) held rigidly constant across all pipelines, with GAT as a secondary sensitivity check only.

**Main research gap:** Reported EEG dementia-classification accuracies span roughly 80–99% on *the same 88-subject dataset*, but the variation tracks preprocessing choices and evaluation protocol rather than model capability — and no study has isolated the representation while holding the graph model fixed. This is not our assertion alone; the 2026 AHEPA benchmark scoping review states that apparent deep-learning superiority "arises not from inherent representational advantage, but from data leakage and overfitting to non-independent samples."

**Expected novelty:** The first controlled, subject-level, leakage-free factorial comparison of EEG signal representations *and* graph-construction rules under a single fixed GNN, on a public dataset, with released code.

**Main risks:** (1) 88 subjects is small — subject-level splits give a test set of only ~18 subjects, so confidence intervals will be wide; (2) a negative result (representation doesn't matter much) is possible — mitigated because a well-powered negative result is itself publishable given the leakage findings above; (3) FTD is the minority class (23 subjects).

---

# Section 2 — Disease Ranking

Scored 1–10 per criterion, weighted as specified in the brief.

| Criterion | Weight | AD+FTD | Parkinson's | Epilepsy | MCI (alone) | MDD | Schizophrenia |
|---|---|---|---|---|---|---|---|
| Public dataset availability | 25% | 9 | 9 | 9 | 5 | 6 | 7 |
| Dataset size | 15% | 6 | 7 | 9 | 4 | 5 | 5 |
| Difficulty / research challenge | 15% | 8 | 7 | 5 | 9 | 7 | 6 |
| Existing SOTA (headroom) | 15% | 7 | 6 | 3 | 7 | 5 | 4 |
| Research gap / GNN opportunity | 15% | 9 | 8 | 5 | 8 | 6 | 6 |
| Clinical/scientific impact | 10% | 9 | 8 | 8 | 9 | 7 | 6 |
| Reproducibility | 5% | 9 | 8 | 7 | 5 | 5 | 6 |
| **Weighted total** | | **8.10** | **7.65** | **6.70** | **6.60** | **5.90** | **5.80** |

### Justification

**1. Alzheimer's + FTD — 8.10.** Wins on the combination the brief asked for, not on popularity. Decisive factors: a CC0, BIDS-formatted, single-protocol dataset needing no application (scored 9); a genuinely hard 3-class problem where AD-vs-FTD discrimination is clinically meaningful and not saturated; and an unusually well-documented gap — a 2026 scoping review devoted entirely to benchmarking this dataset exists, which both proves the dataset's centrality and documents the methodological inconsistency we would resolve.

**Deliberate check against the brief's warning ("Do not simply recommend Alzheimer's because it is popular"):** popularity was *not* scored. AD+FTD would still win if the field were empty, because the ranking is driven by dataset licence, protocol homogeneity, and class structure. Conversely, AD is *penalised* on dataset size (6/10) — Parkinson's ds004584 has more subjects.

**2. Parkinson's — 7.65.** Genuinely close, and better on raw subject count: OpenNeuro **ds004584** has 149 subjects (100 PD / 49 control) at 63 channels, verified from its participants file. It loses on three points: recordings are short (~282 s vs ~802 s mean), the task is binary rather than 3-class, medication state (ON/OFF) is a confound, and its dataset reference is a preprint rather than a peer-reviewed data paper. **Retained as the external-validation cohort** (Section 13).

**3. Epilepsy — 6.70.** Excellent data availability and volume, but rejected on two grounds the brief cares about: seizure detection is close to performance saturation (little headroom to demonstrate a representation effect), and GNNs are already densely applied there. A representation study would be competing against a crowded, mature literature.

**4. MCI alone — 6.60.** Scientifically the most valuable target, but there is no single large, open, standard-protocol MCI EEG corpus; the strong MCI datasets (e.g. CAUEEG) are access-restricted. Best handled as a future extension.

**5–6. MDD (5.90) / Schizophrenia (5.80).** Both suffer from access friction and implausibly high reported accuracies that suggest widespread leakage, making a clean comparison hard to position.

---

# Section 3 — Dataset Ranking

### Rank 1 — OpenNeuro ds004504 (AHEPA / Ioannina) ⭐ RECOMMENDED

Every field below was read directly from the dataset's own files on 17 Aug 2026, not from the paper.

| Field | Verified value | How verified |
|---|---|---|
| Dataset name | "A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects" | `dataset_description.json` |
| Official source | `https://openneuro.org/datasets/ds004504` | — |
| Dataset DOI | **10.18112/openneuro.ds004504.v1.0.8** (resolves, HTTP 200) | doi.org resolution |
| ⚠ DOI discrepancy | The dataset's own `dataset_description.json` records `10.18112/openneuro.ds004504.v1.0.9`, which **does not resolve (HTTP 404)**. Cite v1.0.8. | doi.org resolution |
| Dataset paper | Miltiadous et al. 2023, *Data* 8(6):95 — DOI 10.3390/data8060095 | Crossref |
| Total subjects | **88** | `participants.tsv` |
| AD / FTD / CN | **36 / 23 / 29** | `participants.tsv` — **independently corroborated** by the DICE-Net abstract [F2]: "Recordings of 36 AD, 23 Frontotemporal dementia (FTD), and 29 age-matched healthy individuals (CN) were used." |
| Age (mean±SD) | AD 66.4±7.9; FTD 63.7±8.2; CN 67.9±5.4 | computed from `participants.tsv` |
| MMSE (mean±SD) | AD 17.8±4.5; FTD 22.2±2.6; CN 30.0±0.0 | computed from `participants.tsv` |
| Sex (F/M) | AD 24/12; FTD 9/14; CN 11/18 | computed from `participants.tsv` |
| EEG channels | **19** (Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz) | `channels.tsv` |
| Montage / reference | 10–20; reference A1–A2 | `eeg.json` |
| Sampling frequency | **500 Hz** | `eeg.json` |
| Recording condition | Resting state, **eyes closed** | task label `eyesclosed` |
| Device | Nihon Kohden EEG 2100 | `eeg.json` |
| Applied filter | 0.4–50 Hz; line frequency 50 Hz | `eeg.json` |
| Total duration | **70,590 s = 19.61 h** (AD 490.1 min, FTD 279.2 min, CN 407.2 min) | summed over all 88 `eeg.json` |
| Per-subject duration | mean 802 s; range 307–1291 s | computed |
| Signal format | EEGLAB `.set` (BIDS v1.2.1) | file listing |
| Licence | **CC0** — no application, no DUA | `dataset_description.json` |
| Accessibility | Fully open; direct S3 access without login | verified by direct download |
| Preprocessing standardised? | Partially — a preprocessed derivative is distributed alongside raw | dataset structure |
| Subject-level splitting possible? | **Yes** — one recording per subject, so subject-level splits are trivially enforceable | `participants.tsv` |
| Known leakage issue? | **Yes, in the literature, not the data** — see the AHEPA benchmark finding, Section 17 | 10.1007/s11571-026-10464-w |

**Why rank 1:** it is the only candidate that is simultaneously fully open (CC0, no application), single-site and single-protocol (so a 3-class task needs no dataset merging), multi-class, and already the de-facto benchmark.

**Honest weaknesses:** 88 subjects is small; class imbalance (23 FTD); a single site and single device means no cross-device generalisation claim is possible; unequal recording lengths require care so that longer recordings don't dominate window-level training.

### Rank 2 — BrainLat
Multimodal Latin-American neurodegeneration dataset (AD, FTD, PD, MS), high-density EEG. Paper: 10.1038/s41597-023-02806-8 (*Scientific Data* 2023, 72 citations). Larger and more diverse, **but** requires an access request, and it is multi-site — which introduces exactly the site confound our design tries to avoid. Excellent future external-validation target.

### Rank 3 — CAUEEG (Chung-Ang University Hospital)
Paper: 10.1016/j.neuroimage.2023.120054 (*NeuroImage* 2023, 97 citations). Much larger and includes an MCI class, which ds004504 lacks. Access-restricted, which fails the brief's reproducibility priority.

### Rank 4 — OpenNeuro ds004584 (Parkinson's, Iowa)
**149 subjects (100 PD / 49 control)**, 63 channels, 500 Hz, ~282 s/subject, CC0 — all verified from dataset files. **Recommended as our external-validation cohort.** Its reference is a preprint (10.1101/2022.07.26.22278079), not peer-reviewed.

### Rank 5 — TDBRAIN
10.1038/s41597-022-01409-z (*Scientific Data* 2022, 92 citations). 1,274 subjects across many disorders — large, but psychiatric-heavy and licence-restricted for commercial use; dementia labels are not its focus.

### Rank 6 — OpenNeuro ds002778 (UC San Diego PD)
31 subjects (CC0, 512 Hz, 40 channels). Too small to be primary; useful only as a robustness probe.

### Not recommended for this project
- **TUH EEG Corpus** (10.3389/fnins.2016.00196) — very large but requires application, and is clinically heterogeneous rather than a controlled dementia cohort.
- **Bonn dataset** (10.1103/physreve.64.061907) — 100 single-channel segments per set; single-channel data cannot support a channel-graph at all.
- **CHB-MIT** — paediatric epilepsy; wrong disease per Section 2.

---

# Section 4 — Paper Search (Discovery and Screening)

### Search strategy actually executed

Programmatic queries against the **OpenAlex** API (`title_and_abstract.search`), 37 queries drawn from the brief's Section 23 list plus dataset- and pipeline-specific queries, each returning up to 50 results ranked by citation count, plus a 2025-onward date-filtered sweep to satisfy the recency requirement. Yield: **1,091 unique DOI-bearing works**. Metadata cross-checked against **Crossref**, **DataCite**, **Semantic Scholar**, and **Europe PMC**.

| Query | New records |
|---|---|
| EEG neurological disease classification | 49 |
| EEG Alzheimer classification | 44 |
| EEG Parkinson classification | 41 |
| EEG epilepsy seizure classification | 48 |
| EEG preprocessing neurological disease | 42 |
| EEG signal processing Alzheimer | 40 |
| EEG graph neural network | 47 |
| EEG graph convolutional network classification | 35 |
| EEG functional connectivity disease classification | 43 |
| EEG preprocessing comparison pipeline | 33 |
| EEG representation learning neurological disorder | 47 |
| neurological disorder classification DL limitations | 45 |
| CHB-MIT scalp EEG database seizure | 47 |
| OpenNeuro EEG dataset dementia frontotemporal | 14 |
| EEG wavelet transform feature extraction | 47 |
| EEG spectrogram CNN classification | 45 |
| EEG Hjorth parameters entropy features | 27 |
| EEG transformer classification | 46 |
| EEG channel spatial relationship deep learning | 47 |
| graph attention network brain disease diagnosis | 49 |
| EEG subject independent cross validation leakage | 29 |
| data leakage ML neuroimaging evaluation | 7 |
| *+ 15 further queries incl. 2025+ sweeps* | ~310 |

### Screening funnel

| Stage | Count |
|---|---|
| Harvested (unique DOIs) | 1,091 |
| Passed topical filter (EEG/signal + disease/graph/pipeline) | ~430 |
| Examined in detail (metadata verified; full text sought) | 87 |
| Full text successfully retrieved | 39 |
| **Selected** | **30** |
| Rejected at detailed-examination stage | 57 |

### Papers examined in detail but REJECTED — with reasons

| # | Paper (DOI) | Yr | Reason for rejection |
|---|---|---|---|
| 1 | 10.1109/taffc.2018.2817622 | 2018 | Emotion recognition, not disease. Retained only as graph-construction background. |
| 2 | 10.1109/taffc.2020.2994159 | 2022 | Emotion, not disease classification. |
| 3 | 10.1109/tnsre.2021.3110665 | 2021 | Sleep staging, not neurological disease — though its channel-graph critique is quoted in Section 18. |
| 4 | 10.1016/j.cmpb.2021.106277 | 2021 | Epilepsy — disease not selected. |
| 5 | 10.48550/arXiv.2104.08336 | 2021 | Epilepsy/seizure GNN — disease not selected. |
| 6 | 10.3389/fnins.2016.00196 | 2016 | TUH corpus — dataset not selected (access-restricted). |
| 7 | 10.1103/physreve.64.061907 | 2001 | Bonn dataset — single-channel, cannot form a channel graph. |
| 8 | 10.1109/access.2022.3198988 | 2022 | DWT+ML on AD, but dataset and split protocol not verifiable from available text. |
| 9 | 10.1016/j.bspc.2023.105266 | 2023 | Paywalled; preprocessing detail unverifiable. |
| 10 | 10.1007/s13246-024-01425-w | 2024 | Paywalled; CNN architecture paper, no representation comparison. |
| 11 | 10.1007/s11571-022-09859-2 | 2022 | Hand-crafted pattern model; no graph relevance. |
| 12 | 10.1016/j.clinph.2016.10.002 | 2017 | Superseded by more recent connectivity work on the selected dataset. |
| 13 | 10.3390/diagnostics13030477 | 2023 | Broad AI overview; low methodological specificity. |
| 14 | 10.3389/fnins.2023.1272834 | 2023 | Multi-feature fusion, but no graph/representation contrast. |
| 15 | 10.1038/s41598-025-19727-8 | 2025 | 3-D CNN on neuroimaging, not EEG-graph relevant. |
| 16 | 10.1186/s12911-025-02924-w | 2025 | Uses BrainLat/in-house, not the selected dataset; image-representation only. |
| 17 | 10.1038/s41598-025-12526-1 | 2025 | 3-channel portable device — too few channels for a graph. |
| 18 | 10.3389/fnhum.2025.1526554 | 2025 | PPA/MCI, low-density EEG; different cohort. |
| 19 | 10.1016/j.ymeth.2025.08.003 | 2025 | Ensemble/SVM hybrid; no graph or representation contrast. |
| 20 | 10.1109/access.2025.3590304 | 2025 | GNN for biometric identification, not disease. |
| 21 | 10.1109/access.2026.3683199 | 2026 | Dataset quality framework; useful context, not a classification study. |
| 22 | 10.1002/alz.12645 | 2022 | Review of MCI-to-dementia risk; not an EEG pipeline study. |
| 23 | 10.1016/j.neuroimage.2020.116795 | 2020 | EEG-vs-MRI comparison; different research question. |
| 24 | 10.1088/1741-2552/acb96e | 2023 | SCD/MCI cohort; dataset not available to us. |
| 25 | 10.3390/e20010035 | 2018 | Superseded by 2025–26 work on the same problem. |
| 26 | 10.1186/s12911-018-0613-y | 2018 | Superseded; private dataset. |
| 27 | 10.1109/tnsre.2019.2909100 | 2019 | Private dataset; not reproducible. |
| 28 | 10.1038/s41598-023-32664-8 | 2023 | Methods survey; largely descriptive. |
| 29 | 10.1186/s13195-024-01582-w | 2024 | Biomarker study, not a classification pipeline. |
| 30 | 10.1109/access.2025.3585196 | 2025 | MST analysis; narrow single-metric scope. |
| 31 | 10.3389/fnins.2025.1555657 (kept) / near-duplicates | 2025 | Several near-duplicate GCN-on-ds004504 papers dropped in favour of the strongest. |
| 32 | 10.29284/bsf3af03 | 2025 | **Predatory/unverifiable venue** — excluded per Section 4 of brief. |
| 33 | 10.25258/ijddt.16.28s.106 | 2026 | **Predatory venue** (drug-delivery journal publishing an EEG-GNN paper). |
| 34 | 10.52783/jisem.v10i2s.210 | 2025 | **Predatory venue.** |
| 35 | 10.55041/ijsrem54079 | 2025 | **Non-peer-reviewed.** |
| 36 | 10.5281/zenodo.21546726 | 2025 | **Preprint/repository deposit, not peer-reviewed.** |
| 37 | 10.6084/m9.figshare.32560329 | 2026 | **Figshare deposit, not peer-reviewed.** |
| 38 | 10.53759/7669/jmc202505117 | 2025 | **Low-quality venue.** |
| 39 | 10.57197/jdr-2025-0633 / -0590 | 2025 | Low methodological detail. |
| 40 | 10.14569/ijacsa.2025.0160590 | 2025 | Weak methodology; leakage-prone evaluation. |
| 41 | 10.70917/ijcisim-2026-3951 | 2026 | GUI/engineering paper, not classification. |
| 42 | 10.5772/acrt20250147 | 2026 | Dyslexia; different disorder. |
| 43 | 10.1002/alz70856_105594, 10.1002/alz70860_104123 | 2025 | Conference abstracts, not full papers. |
| 44 | 10.12732/ijam.v38i6s.417 | 2025 | Mathematics venue; unverifiable protocol. |
| 45 | 10.32598/cjns.11.42.461.3 | 2025 | Regional journal; small sample, no code. |
| 46–57 | *(various emotion/BCI/motor-imagery GNN papers)* | — | Graph methods but non-clinical task; used only as architectural background. |

---

# Section 5 — Final 30 Papers

Citation counts: **CR** = Crossref, **OA** = OpenAlex, **S2** = Semantic Scholar. All checked **17 Aug 2026**. All DOIs verified as resolving.

> **Category key (per brief Section 24)** — **A** = authors explicitly name GNN/graph neural networks as future work; **B** = authors explicitly recommend graph/network/connectivity modelling but do not say "GNN"; **C** = no graph mention, but a stated limitation strongly motivates graph modelling; **D** = already uses GNNs.
>
> **These categories are kept strictly separate and are never merged.** A paper is placed in A **only** where a verbatim quotation was extracted from its retrieved full text. Where full text could not be retrieved, the category is marked *"unclassified — full text not retrievable"* rather than guessed.

## 5.1 — The 2025–2026 cohort (16 papers)

---
**[R1] The AHEPA EEG benchmark: setting the standard for machine learning in dementia diagnosis, a scoping review**
1. Citation: Miltiadous A., Ntetska A., Aspiotis V., Moustakli E., Tsipouras M.G., Tzallas A.T., Giannakeas N., Glavas E., Angelidis P., Tzimourta K.D. (2026)
2. Year: 2026 · 3. Venue: *Cognitive Neurodynamics* 20(1), Springer
4. DOI: `10.1007/s11571-026-10464-w` · 5. Citations: CR 0 / OA 0 / **S2 1** (new)
6. Dataset: ds004504 (review of all studies using it) · 7. Subjects: 88 · 8. Signal: scalp EEG
9. Disease: AD, FTD · 10. Task: scoping review of AD/FTD/CN classification
11. Preprocessing: surveys all pipelines in the literature · 12. Model: surveys ML/CNN/RNN/GNN/Transformer
13. Results: reports convergence at ≈80–85% for AD vs CN under sound protocols
14. Limitation identified: widespread leakage and non-independent sampling across the literature
15. GNN mentioned? **Yes** · 16. Category: **D + primary gap evidence**
17. **GNN Future-Work Evidence**
 - *Section:* Discussion · *Page:* n/a (HTML full text, Europe PMC PMC13184051)
 - *Quotation:* "much of the apparent superiority of deep learning arises not from inherent representational advantage, but from data leakage and overfitting to non-independent samples."
 - *Second quotation:* "Graph neural networks exploit the topological structure of EEG connectivity, modeling long-range dependencies that CNNs cannot easily capture."
 - *Third quotation:* "these extensions do not demonstrate a systematic performance improvement, with all approaches converging to similar accuracy levels (≈ 80–85% for AD vs. …)"
 - *Why this supports our direction:* This is the single strongest justification for the whole project. An independent 2026 review of our exact dataset states that (a) reported gains are confounded by leakage, (b) added representations have **not** been systematically evaluated, and (c) graph models are a genuine conceptual advance. Our controlled design directly answers all three.
18. Relevance: **10/10**

---
**[R2] Deep Learning-Based Alzheimer's Disease Detection from Multi-Channel EEG Using Fused Time–Frequency Representations**
1. Citation: (2026) · 2. Year: 2026 · 3. Venue: *Diagnostics* 16(5):746, MDPI
4. DOI: `10.3390/diagnostics16050746` · 5. Citations: CR 1 / OA 1 / S2 1
6. Dataset: ds004504 · 7. Subjects: 88 · 8. Signal: scalp EEG, 19 ch
9. Disease: AD/dementia · 10. Task: dementia detection
11. Preprocessing: five time-frequency representations (STFT, CWT, WVD, HHT, CQT) compared under one unified DL setup; 4×5 grid channel fusion into a composite image
12. Features: time-frequency images · 13. Model: pretrained CNNs (InceptionV3 best)
14. Results: STFT best; CWT and CQT comparable; WVD weakest — "classification performance is influenced by the choice of representation"
15. Limitation: fixed grid fusion does not model electrode topology
16. Category: **B** (explicit graph recommendation, does not use the term GNN)
17. **GNN Future-Work Evidence**
 - *Section:* 4.4 Limitations, and 5 Conclusions · *Page:* n/a (Europe PMC full text)
 - *Quotation:* "the proposed 4 × 5 fixed grid fusion strategy, while effective for integrating all 19 EEG channels into a unified image representation, **does not explicitly model the true spatial topology or anatomical neighborhood relationships among electrodes on the scalp**. As a result, spatial dependencies are implicitly learned by the convolutional networks rather than being explicitly encoded, and **alternative graph-based or topology-aware fusion schemes may further enhance spatial interpretability and performance**."
 - *Second quotation:* "Future research will address these limitations by … exploring **topology-aware fusion mechanisms** and subject-level modeling strategies."
 - *Why this supports our direction:* This is the closest existing work to our proposal and it validates the design while leaving our contribution open. They performed a controlled representation comparison on our exact dataset — with a CNN — and concluded both that representation choice matters and that the missing ingredient is explicit graph/topology modelling. Our study substitutes a GNN for their CNN and adds the connectivity-based representations they could not encode.
18. Relevance: **10/10**

---
**[R3] Quantitative EEG signatures of power and functional connectivity alterations in Alzheimer's disease and frontotemporal dementia**
2. Year: 2026 · 3. Venue: *Scientific Reports*, Nature Portfolio
4. DOI: `10.1038/s41598-026-42452-9` · 5. Citations: CR 0 / OA 0 / S2 1
6. Dataset: ds004504 · 7. Subjects: 88 · 9. Disease: AD, FTD · 10. Task: biomarker characterisation (AD/FTD/CN)
11. Preprocessing: band-wise spectral power + functional connectivity, lobe-level
13. Model: statistical analysis (not a classifier)
14. Results: AD shows widespread FC disruption; FTD intermediate between AD and CN; occipital alpha/beta power reduced in both
15. Limitations (verbatim topics): single public dataset, no follow-up recordings, no medication data, needs validation in diverse populations
16. Category: **C** — establishes that connectivity carries subtype-discriminative signal but never models it as a learned graph
17. Evidence: "Future work" section is present but concerns **sex differences in FC**, not graph learning: "In the future, we will study the FC patterns between men and women for AD and FTD." *No GNN future-work claim is made — this paper is therefore NOT placed in Category A.*
18. Relevance: 8/10

---
**[R4] Explainable EEG-based machine learning for early diagnosis of Alzheimer's disease and frontotemporal dementia**
2. Year: 2026 · 3. Venue: *Scientific Reports* · 4. DOI: `10.1038/s41598-026-57069-1`
5. Citations: CR 0 / OA 0 / S2 0 · 6. Dataset: ds004504 · 7. Subjects: 88
10. Task: AD/FTD/CN · 13. Model: interpretable ML · 16. Category: **C**
17. Evidence: full text retrieved (OA PDF); no explicit graph/GNN future-work statement found. Recorded honestly as Category C.
18. Relevance: 7/10

---
**[R5] Alzheimer's Disease Detection Using Combined EEG Source Connectivity and Microstate Features**
2. Year: 2026 · 3. Venue: *Brain Sciences* 16(8):856 · 4. DOI: `10.3390/brainsci16080856`
5. Citations: CR 0 / OA 0 / S2 0 · 9. Disease: AD · 10. Task: AD detection
11. Preprocessing: source-space connectivity + microstate features · 16. Category: **B**
17. Evidence: **Full text not retrievable** at time of writing — abstract confirms source-connectivity features are central, but no future-work quotation can be supplied. *Marked "Not verified"; not counted toward Category A.*
18. Relevance: 7/10

---
**[R6] RGB-Style Input Representations for EEG: Evaluating Spatial Concatenation Versus Band-Wise Stacking**
2. Year: 2026 · 3. Venue: *Brain Sciences* 16(7):716 · 4. DOI: `10.3390/brainsci16070716`
5. Citations: CR 0 / OA 0 / S2 0 · 8. Signal: EEG · 10. Task: representation comparison
11. Preprocessing: two competing image-encoding schemes compared directly · 16. Category: **C**
17. Evidence: **Full text not retrievable.** Included because its title and abstract establish, independently, that *how the signal is arranged before the network sees it* is treated as a first-class research question in 2026 — the premise of our study.
18. Relevance: 8/10

---
**[R7] Comparison of preprocessing techniques for effective cognitive analysis using electroencephalography**
2. Year: 2026 · 3. Venue: *MethodsX*, Elsevier · 4. DOI: `10.1016/j.mex.2026.103821`
5. Citations: CR 0 / OA 0 / S2 0 · 10. Task: preprocessing comparison · 16. Category: **C**
17. Evidence (Limitations, verbatim): "The study focuses on signal acquisition from four electrode positions that limits spatial coverage. However, if a study requires broader neurophysiological interpretation from other lobes … there should be a coverage of all electrode positions."
 *Why relevant:* a 2026 paper explicitly comparing preprocessing techniques, whose own stated limitation is insufficient spatial coverage — precisely the axis a 19-channel graph exploits.
18. Relevance: 8/10

---
**[R8] Translational approach for dementia subtype classification using convolutional neural network based on EEG connectome dynamics** ⭐ **CATEGORY A**
2. Year: 2025 · 3. Venue: *Scientific Reports* · 4. DOI: `10.1038/s41598-025-02018-7`
5. Citations: CR 9 / OA 9 / S2 9 · 6. Dataset: EEG connectome dynamics, dementia subtypes
9. Disease: AD, FTD/dementia subtypes · 10. Task: subtype classification
11. Preprocessing: wPLI-based connectivity across frequency bands · 13. Model: CNN
15. Limitation: interactions between brain regions not modelled as a network
16. Category: **A — EXPLICIT GNN FUTURE WORK**
17. **GNN Future-Work Evidence**
 - *Section:* Discussion · *Page:* n/a (Europe PMC full text, PMC12089592)
 - *Quotation:* "These findings highlight the necessity for further investigation to examine the connectivity dynamics as a network. **Future studies should consider employing novel techniques, such as graph convolutional networks**, topological models such as N-body orbital graphs or graphical features **to explore the complex interactions between brain regions**."
 - *Why this supports our direction:* An unambiguous, authors' own recommendation that dementia-subtype classification from EEG connectivity should next be done with graph convolutional networks. This is exactly our proposed model class, on exactly our proposed task.
18. Relevance: **10/10**

---
**[R9] A multi-graph convolutional network method for Alzheimer's disease diagnosis based on multi-frequency EEG data**
2. Year: 2025 · 3. Venue: *Frontiers in Neuroscience* · 4. DOI: `10.3389/fnins.2025.1555657`
5. Citations: CR 3 / OA 4 / S2 5 · 6. Dataset: OpenNeuro (ds004504) · 10. Task: AD diagnosis
11. Preprocessing: multi-frequency band decomposition; functional + structural graphs
13. Model: MF-MGCN (multi-frequency multi-graph GCN) · 16. Category: **D**
17. **Evidence of the gap we exploit** (Introduction, Europe PMC PMC12263931):
 - "However, most studies **overlook spatial connectivity between channels**, which can reveal synchronized activity patterns between different brain regions."
 - "**Most existing research relies solely on functional connectivity methods** to infer inter-regional brain connectivity, **failing to fully capture the complex interactions** within EEG signals."
 - *Why relevant:* A 2025 GNN paper on our dataset states that the choice between spatial and functional edge definitions is unresolved — which is precisely our Factor B (Section 13).
18. Relevance: 9/10

---
**[R10] Connectogram-COH: A Coherence-Based Time-Graph Representation for EEG-Based Alzheimer's Disease Detection**
2. Year: 2025 · 3. Venue: *Diagnostics* 15(11):1441 · 4. DOI: `10.3390/diagnostics15111441`
5. Citations: CR 5 / OA 6 / S2 6 · 6. Dataset: ds004504 · 10. Task: AD detection
11. Preprocessing: coherence-based dynamic connectivity rendered as a time-graph image
13. Model: CNNs over graph-derived images · 16. Category: **D/B**
17. Evidence (Introduction, PMC12154050): "(2021) introduced graph neural networks (GNNs), **which respect the spatial relationships between electrodes and incorporate structural information into learning**, allowing models to exploit topological features of the brain network — **a significant advancement over CNNs, which assume regular grid-like inputs**."
 Also reports a practical finding we adopt: "limiting the segment length values to 10 or 20 s would give better accuracy."
18. Relevance: 9/10

---
**[R11] Graph Spectral Analysis Using Electroencephalography in Alzheimer Disease and Frontotemporal Dementia**
2. Year: 2025 · 3. Venue: *International Journal of Neural Systems* · 4. DOI: `10.1142/s0129065725500480`
5. Citations: CR 5 / OA 5 / S2 0 · 9. Disease: AD, FTD · 16. Category: **D**
17. Evidence: full text not retrievable (paywalled). Metadata verified via Crossref. Included for its graph-spectral treatment of our exact disease pair; **no quotation claimed**.
18. Relevance: 8/10

---
**[R12] Multi-frequency EEG and multi-functional connectivity graph convolutional network based detection**
2. Year: 2025 · 3. Venue: *Complex & Intelligent Systems*, Springer · 4. DOI: `10.1007/s40747-025-01974-x`
5. Citations: CR 8 / OA 6 / S2 8 · 10. Task: disease detection · 13. Model: GCN over multiple connectivity measures
16. Category: **D** — directly relevant because it varies *the connectivity measure* while holding the GCN fixed, a one-factor version of our design.
18. Relevance: 9/10

---
**[R13] Identification of Alzheimer's disease brain networks based on EEG phase synchronization**
2. Year: 2025 · 3. Venue: *BioMedical Engineering OnLine* · 4. DOI: `10.1186/s12938-025-01361-0`
5. Citations: CR 16 / OA 14 / S2 17 · 10. Task: AD identification
11. Preprocessing: phase synchronisation → functional brain network · 16. Category: **B**
17. Evidence (Introduction, PMC11892187): "EEG phase synchronization is used through **the combination of brain network and deep learning** in order to achieve the purpose of assisted diagnosis."
18. Relevance: 8/10

---
**[R14] Classification for Alzheimer's disease and frontotemporal dementia via resting-state electroencephalography**
2. Year: 2025 · 3. Venue: *Cognitive Neurodynamics* · 4. DOI: `10.1007/s11571-025-10232-2`
5. Citations: CR 19 / OA 19 / S2 15 — highest-cited 2025 paper in our set
6. Dataset: ds004504 · 10. Task: AD vs FTD vs CN · 16. Category: **C**
17. Evidence: full text not retrievable. Serves as a **SOTA baseline** (Section 17). No quotation claimed.
18. Relevance: 9/10

---
**[R15] Evaluating EEG complexity and spectral signatures in Alzheimer's disease and frontotemporal dementia**
2. Year: 2025 · 3. Venue: *npj Aging*, Nature Portfolio · 4. DOI: `10.1038/s41514-025-00243-y`
5. Citations: CR 18 / OA 14 / S2 19 · 6. Dataset: ds004504 · 16. Category: **C**
17. Evidence (Introduction, PMC12149297): "In healthy brain networks, EEG signals exhibit scaling behavior… These findings align with broader principles of complexity loss in neurodegenerative disorders, where **brain networks exhibit reduced adaptability and diminished hierarchical organization**." Establishes complexity/entropy features (our Pipeline B) as neurophysiologically grounded.
18. Relevance: 8/10

---
**[R16] Multi-band Morlet mutual information functional connectivity for classifying Alzheimer's disease**
2. Year: 2025 · 3. Venue: *Computers in Biology and Medicine*, Elsevier · 4. DOI: `10.1016/j.compbiomed.2025.111041`
5. Citations: CR 2 / OA 2 / S2 2 · 10. Task: AD classification
11. Preprocessing: Morlet wavelet + mutual-information connectivity, band-wise — combines our Pipelines D and E
16. Category: **B** · 18. Relevance: 9/10

## 5.2 — Foundational and high-impact cohort (14 papers)

---
**[F1] A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG**
Miltiadous et al., 2023, *Data* 8(6):95 · DOI `10.3390/data8060095`
Citations: **CR 263 / OA 259 / S2 281** · Dataset origin paper for ds004504 (full metadata in Section 3/7). Relevance 10/10.

---
**[F2] DICE-Net: A Novel Convolution-Transformer Architecture for Alzheimer Detection in EEG Signals**
Miltiadous et al., 2023, *IEEE Access* · DOI `10.1109/access.2023.3294618`
Citations: **CR 177 / OA 187 / S2 172** · The reference baseline on ds004504; convolution+transformer over spectral/coherence features. Category **C**. Relevance 9/10.

---
**[F3] EEG-Based Graph Neural Network Classification of Alzheimer's Disease: An Empirical Evaluation of Functional Connectivity Methods**
Klepl et al., 2022, *IEEE TNSRE* · DOI `10.1109/tnsre.2022.3204913`
Citations: **CR 132 / OA 126 / S2 123** · Category **D**.
*Most important precedent in the entire review:* it empirically compares functional-connectivity methods as GNN inputs. It is the one-factor (edge-only) ancestor of our two-factor design, and confirms the question is live and publishable. Relevance **10/10**.

---
**[F4] Adaptive Gated Graph Convolutional Network for Explainable Diagnosis of Alzheimer's Disease Using EEG Data**
Klepl et al., 2023, *IEEE TNSRE* · DOI `10.1109/tnsre.2023.3321634`
Citations: CR 46 / OA 46 / S2 54 · Category **D**. Learned adjacency + explainability; a strong comparison target. Relevance 9/10.

---
**[F5] Graph Neural Network-Based EEG Classification: A Survey**
Klepl et al., 2024, *IEEE TNSRE* · DOI `10.1109/tnsre.2024.3355750`
Citations: **CR 125 / OA 116 / S2 101** · Category **D**. The definitive taxonomy of node/edge/feature choices for EEG graphs; our graph-construction decisions in Section 11 follow its vocabulary. Relevance 10/10.

---
**[F6] Spatial–temporal graph convolutional network for Alzheimer classification based on brain functional connectivity**
Shan et al., 2022, *Human Brain Mapping* · DOI `10.1002/hbm.25994`
Citations: CR 70 / OA 74 / S2 69 · Category **D**.
*Evidence (Introduction, PMC9812255):* "The above CNN applications on AD classification … focus more on learning the locally and continuously changed multiscaled features **on the Euclidean space** from the EEG signals, **neglecting the functional connectivity features**." And: "**Instead, geometric graph-based deep learning methods would provide a more suitable way to learn the cross-channel topologically associated features of EEG.**" Relevance 9/10.

---
**[F7] An interpretable model based on graph learning for diagnosis of Parkinson's disease with voice-related EEG** ⭐ **CATEGORY A**
2024, *npj Digital Medicine* · DOI `10.1038/s41746-023-00983-9`
Citations: CR 50 / OA 48 / S2 49 · Category **D + A**
**GNN Future-Work Evidence** — *Section:* Discussion (limitations), Europe PMC PMC10770376:
"Also, **the exclusive use of GCN models in the present study may not be suitable for all types of graph data. Future studies should consider alternative methodologies (e.g. graph attention networks, graph neural ordinary differential equations) for graph learning.**" Also: "our models performed the connectivity analysis of large-scale EEG networks **at the electrode level, which cannot provide precise anatomical sources**."
*Why this supports our direction:* the authors of a successful GCN paper explicitly call for systematic comparison across graph architectures — supporting our GAT sensitivity analysis — and flag electrode-level vs source-level graphs as unresolved. Relevance 9/10.

---
**[F8] Leakage and the reproducibility crisis in machine-learning-based science**
Kapoor & Narayanan, 2023, *Patterns* · DOI `10.1016/j.patter.2023.100804`
Citations: **CR 812 / OA 840 / S2 860** · The methodological backbone for our Section 16 leakage controls. Relevance 9/10.

---
**[F9] Functional neural network analysis in frontotemporal dementia and Alzheimer's disease using EEG and graph theory**
de Haan et al., 2009, *BMC Neuroscience* · DOI `10.1186/1471-2202-10-101`
Citations: **CR 333 / OA 408 / S2 406** · Category **B**
*Evidence — dedicated "Future directions" section (PMC2736175):* "**Graph theory offers a growing amount of techniques to describe topological network features** like modularity, node centrality (e.g. 'betweenness'), or synchronizability." Also: "An alternative approach is to convert the original SL-based connectivity matrix directly into a **'weighted' graph**, in which the connections between nodes in a graph have variable strengths."
*Why relevant:* the seminal demonstration that AD and FTD produce *dissimilar* network disturbances — the neurophysiological premise of a 3-class graph model — with an explicit call for weighted-graph methods. Relevance 9/10.

---
**[F10] A novel multi-modal machine learning based approach for automatic classification of EEG recordings in dementia**
2020, *Neural Networks* · DOI `10.1016/j.neunet.2019.12.006`
Citations: **CR 305 / OA 333 / S2 287** · Category **C**. Highest-cited modern dementia-EEG classification paper; the multi-modal feature baseline. Relevance 8/10.

---
**[F11] Phase lag index: assessment of functional connectivity from multi-channel EEG and MEG with diminished bias from common sources**
Stam et al., 2007, *Human Brain Mapping* · DOI `10.1002/hbm.20346`
Citations: **CR 1,896 / OA 2,088 / S2 2,015** · Methodological basis for our volume-conduction-robust edge definition (wPLI/PLI). Relevance 9/10.

---
**[F12] Small-World Networks and Functional Connectivity in Alzheimer's Disease**
Stam et al., 2006, *Cerebral Cortex* · DOI `10.1093/cercor/bhj127`
Citations: **CR 1,083 / OA 1,292 / S2 1,297** · Category **B**. Establishes that AD is a *network* disorder — the biological warrant for graph modelling. Relevance 9/10.

---
**[F13] Semi-Supervised Classification with Graph Convolutional Networks**
Kipf & Welling, 2016, arXiv (ICLR 2017) · DOI `10.48550/arXiv.1609.02907` (verified via DataCite)
Citations: OA 8,058 (Crossref n/a — arXiv DOIs are DataCite-registered)
The architecture we recommend as the fixed downstream model. Relevance 10/10.

---
**[F14] EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces**
Lawhern et al., 2018, *J. Neural Engineering* · DOI `10.1088/1741-2552/aace8c`
Citations: **CR 4,585 / OA 4,537 / S2 4,638** · Our required non-graph deep baseline (Section 17). Relevance 8/10.

---

# Section 6 — 2025–2026 Papers (Mandatory Requirement)

**Requirement:** ≥15 of the final 30 published in 2025 or 2026. **Status: SATISFIED — 16 of 30 (53%).**

| # | DOI | Year | Venue | Publisher class |
|---|---|---|---|---|
| R1 | 10.1007/s11571-026-10464-w | **2026** | Cognitive Neurodynamics | Springer |
| R2 | 10.3390/diagnostics16050746 | **2026** | Diagnostics | MDPI (strong, on-topic) |
| R3 | 10.1038/s41598-026-42452-9 | **2026** | Scientific Reports | Nature Portfolio |
| R4 | 10.1038/s41598-026-57069-1 | **2026** | Scientific Reports | Nature Portfolio |
| R5 | 10.3390/brainsci16080856 | **2026** | Brain Sciences | MDPI |
| R6 | 10.3390/brainsci16070716 | **2026** | Brain Sciences | MDPI |
| R7 | 10.1016/j.mex.2026.103821 | **2026** | MethodsX | Elsevier |
| R8 | 10.1038/s41598-025-02018-7 | 2025 | Scientific Reports | Nature Portfolio |
| R9 | 10.3389/fnins.2025.1555657 | 2025 | Frontiers in Neuroscience | Frontiers |
| R10 | 10.3390/diagnostics15111441 | 2025 | Diagnostics | MDPI |
| R11 | 10.1142/s0129065725500480 | 2025 | Int. J. Neural Systems | World Scientific |
| R12 | 10.1007/s40747-025-01974-x | 2025 | Complex & Intelligent Systems | Springer |
| R13 | 10.1186/s12938-025-01361-0 | 2025 | BioMedical Engineering OnLine | BMC |
| R14 | 10.1007/s11571-025-10232-2 | 2025 | Cognitive Neurodynamics | Springer |
| R15 | 10.1038/s41514-025-00243-y | 2025 | npj Aging | Nature Portfolio |
| R16 | 10.1016/j.compbiomed.2025.111041 | 2025 | Computers in Biology and Medicine | Elsevier |

**Honest note on the recency requirement.** Enough genuinely high-quality 2025–2026 work exists, so no quota-filling was needed. However, a large volume of 2025–2026 EEG-GNN output sits in predatory or non-peer-reviewed venues (see rejection table rows 32–38); those were excluded even though including them would have made the quota trivially easy. Note also that 2026 papers necessarily have near-zero citation counts — they are included on **scientific relevance**, judged by venue quality and methodological content, exactly as the brief's citation-velocity guidance requires.

---

# Section 7 — Dataset Papers (Origin Papers)

| Dataset | Origin paper | DOI | Verified | Citations (CR/OA/S2) |
|---|---|---|---|---|
| **ds004504 (AHEPA)** ⭐ | Miltiadous et al. 2023, *Data* 8(6):95 | `10.3390/data8060095` | ✓ Crossref | 263 / 259 / 281 |
| ds004504 dataset record | OpenNeuro | `10.18112/openneuro.ds004504.v1.0.8` | ✓ resolves (v1.0.9 → 404) | n/a |
| ds004584 (PD, Iowa) | preprint | `10.1101/2022.07.26.22278079` | ✓ (preprint) | n/a |
| ds004584 dataset record | OpenNeuro | `10.18112/openneuro.ds004584.v1.0.0` | ✓ resolves (302) | n/a |
| ds002778 (PD, UCSD) | OpenNeuro record | `10.18112/openneuro.ds002778.v1.0.4` | ✓ | n/a |
| BrainLat | Prado et al. 2023, *Scientific Data* | `10.1038/s41597-023-02806-8` | ✓ Crossref | 72 / — / — |
| CAUEEG | Kim et al. 2023, *NeuroImage* | `10.1016/j.neuroimage.2023.120054` | ✓ Crossref | 97 / — / — |
| TDBRAIN | van Dijk et al. 2022, *Scientific Data* | `10.1038/s41597-022-01409-z` | ✓ Crossref | 92 / — / — |
| TUH EEG Corpus | Obeid & Picone 2016, *Front. Neurosci.* | `10.3389/fnins.2016.00196` | ✓ Crossref | 479 / 521 / 605 |
| Bonn EEG | Andrzejak et al. 2001, *Phys. Rev. E* | `10.1103/physreve.64.061907` | ✓ Crossref | 2,439 / 3,013 / 2,954 |
| CHB-MIT (host) | Goldberger et al. 2000, *Circulation* (PhysioNet) | `10.1161/01.cir.101.23.e215` | ✓ Crossref | — |
| CHB-MIT (origin) | Shoeb 2009, MIT PhD thesis | **DOI: Not available** | — | 860 (OA) |

**Supporting methods papers with verified DOIs:** EEGLAB `10.1016/j.jneumeth.2003.10.009` (CR 23,070); MNE `10.1016/j.neuroimage.2013.10.027` (CR 1,979); ICLabel `10.1016/j.neuroimage.2019.05.026` (CR 2,343); autoreject `10.1016/j.neuroimage.2017.06.030` (CR 632); PREP pipeline `10.3389/fninf.2015.00016` (CR 1,172); DISCOVER-EEG `10.1038/s41597-023-02525-0` (CR 66).

---

# Section 8 — Preprocessing Pipeline Taxonomy

**Signal → Processing → Representation → Graph → GNN**, as observed across the reviewed literature.

```
RAW EEG  (88 subjects × 19 ch × 500 Hz)
   │
   ├─ STAGE 1  FILTERING
   │    band-pass 0.5–45 Hz · notch 50 Hz · [dataset ships 0.4–50 Hz applied]
   │
   ├─ STAGE 2  ARTIFACT HANDLING
   │    ICA + ICLabel · ASR · autoreject · re-reference (avg / A1-A2)
   │
   ├─ STAGE 3  SEGMENTATION
   │    fixed windows (literature converges on 10–20 s; [R10])
   │
   ├─ STAGE 4  TRANSFORMATION ───────────── the axis we vary ──────────────┐
   │    (a) none/raw            (b) time-domain stats                      │
   │    (c) FFT/PSD/Welch       (d) STFT / CWT / DWT / WPT / CQT / HHT     │
   │    (e) connectivity: COH, PLV, PLI, wPLI, MI, Granger                 │
   │    (f) image-like: spectrogram, scalogram, topomap, RP, GAF          │
   │                                                                       │
   ├─ STAGE 5  NUMERICAL REPRESENTATION ────────────────────────────────────┤
   │    node-feature matrix X ∈ ℝ^(19×F)   and/or   adjacency A ∈ ℝ^(19×19)│
   │                                                                       │
   ├─ STAGE 6  GRAPH CONSTRUCTION ─────────────────────────────────────────┘
   │    nodes = channels · edges = spatial | functional | learned | hybrid
   │    sparsification = threshold | kNN | MST · normalisation = D^-½(A+I)D^-½
   │
   └─ STAGE 7  MODEL:  GCN / GAT / ChebNet / GraphSAGE / GIN / ST-GCN
```

**Observed mapping from literature to this taxonomy**

| Branch | Representative selected papers | Dominant model used | Graph used? |
|---|---|---|---|
| (a) Raw/minimal | EEGNet [F14], DICE-Net [F2] | CNN / Transformer | No |
| (b) Time-domain | [R15] complexity/entropy | ML | No |
| (c) Frequency | [R3], [F2], [F10] | statistics / ML / CNN | No |
| (d) Time-frequency | **[R2]** (5 TFRs compared), [R16] | CNN | No → recommends graphs |
| (e) Connectivity | [R8], [R13], [R16], [F3], [F9], [F11], [F12] | ML / GCN | Yes |
| (f) Image-like | [R2], [R6], [R10] | CNN | Partly (as image) |
| Fused (d+e) | [R16], [R9], [R12] | GCN | Yes |

**The structural finding.** Branches (a)–(d) and (f) dominate the dementia literature and almost never use graphs; branch (e) uses graphs but almost never varies the node-feature representation. **No selected paper varies both axes under one fixed model.** That intersection is the gap.

---

# Section 9 — Recommended Preprocessing Experiments

Six pipelines, ranked by expected scientific value. All share Stages 1–3 **identically** — this is what makes the comparison controlled.

**Fixed preamble for every pipeline (identical code path):**
`raw .set → band-pass 0.5–45 Hz → notch 50 Hz → average reference → ICA(ICLabel, drop eye/muscle ≥0.8) → resample 128 Hz → 10 s non-overlapping windows`
→ yields ≈ 7,059 windows total (70,590 s ÷ 10 s), each `(19 ch × 1,280 samples)`.

---
### P1 — Frequency-domain band power  *(rank 1: best value/complexity ratio)*
```
window (19×1280) → Welch PSD → 5 bands (δ 0.5-4, θ 4-8, α 8-13, β 13-30, γ 30-45)
  → relative band power + spectral entropy + peak freq   → X ∈ ℝ^(19×7)
  → A from wPLI (α band), kNN k=4, sym-normalised        → A ∈ ℝ^(19×19)
  → GCN
```
Scientifically valid (spectral slowing is the best-established AD marker); cheapest; retains band-specific power; discards phase and temporal dynamics; fully reproducible; ideal GNN compatibility (compact node features); leakage risk low; difficulty low.

### P2 — Connectivity-only  *(rank 2: the graph-native pipeline)*
```
window → band-pass per band → wPLI/PLV/coherence per band → A ∈ ℝ^(19×19) per band
  → X = node degree/strength/clustering (or identity)     → X ∈ ℝ^(19×5)
  → GCN
```
The only pipeline where information lives entirely in the edges. Retains inter-channel synchrony; discards amplitude. This is the pipeline the connectivity literature ([F3], [F11], [F12]) supports most directly.

### P3 — Time-frequency (wavelet) node features  *(rank 3)*
```
window → CWT (Morlet) or DWT (db4, 5 levels) → per-channel sub-band energy,
  entropy, mean, std → X ∈ ℝ^(19×20) → A from wPLI → GCN
```
Retains non-stationary structure that PSD averages away; [R2] found STFT/CWT strongest among TFRs, [R16] pairs Morlet with MI connectivity.

### P4 — Time-domain statistical  *(rank 4: the honest cheap baseline)*
```
window → mean, variance, RMS, skewness, kurtosis, ZCR,
  Hjorth activity/mobility/complexity, sample entropy → X ∈ ℝ^(19×10) → A from wPLI → GCN
```
Very cheap and interpretable; discards all spectral detail. Included because if it matches P1/P3, that is a major (and publishable) negative result.

### P5 — Raw/minimal  *(rank 5: the lower bound)*
```
window → z-score per channel per window → X ∈ ℝ^(19×1280) (or 1D-CNN encoder → ℝ^(19×64))
  → A from wPLI → GCN
```
Discards nothing, but forces the GNN to learn temporal structure from 88 subjects — expected to underperform. Establishes the floor.

### P6 — Fused (frequency + connectivity + time-frequency)  *(rank 6 by cost, likely rank 1 by accuracy)*
```
X = concat(P1 features, P3 features, node graph metrics) → X ∈ ℝ^(19×~34)
  → A from wPLI → GCN
```
Tests whether representations are complementary or redundant — directly answers RQ3.

### Pipeline comparison matrix

| | P1 Freq | P2 Conn | P3 TF | P4 Time | P5 Raw | P6 Fused |
|---|---|---|---|---|---|---|
| Node feature dim | 7 | 5 | 20 | 10 | 1280/64 | ~34 |
| Info retained | band power | phase synchrony | non-stationarity | waveform shape | everything | most |
| Info discarded | phase, time | amplitude | fine timing | spectrum | nothing | little |
| Compute cost | Low | Medium | Medium-High | Low | High | High |
| GNN compatibility | Excellent | Excellent | Good | Excellent | Poor-Fair | Good |
| Reproducibility | Excellent | Good (measure-sensitive) | Good | Excellent | Excellent | Fair |
| Leakage risk | Low | **Medium** — connectivity over long windows can straddle splits | Low | Low | Low | Medium |
| Expected difficulty | Low | Medium | Medium | Low | High | Medium |

---

# Section 10 — GNN Recommendation

**Recommended primary architecture: a 2-layer GCN (Kipf & Welling, `10.48550/arXiv.1609.02907`), held constant across all pipelines.**

```
X ∈ ℝ^(19×F), Â = D̃^-½(A+I)D̃^-½
H1 = ReLU(Â X W0)        W0 ∈ ℝ^(F×64)    + BatchNorm + Dropout(0.5)
H2 = ReLU(Â H1 W1)       W1 ∈ ℝ^(64×32)
g  = mean-pool over 19 nodes  → ℝ^32  → Linear(32→3) → softmax
```
≈ (F×64 + 64×32 + 32×3) ≈ 2.6k–3.6k parameters for F=7–20.

### Justification

1. **It is the correct control.** The independent variable is the representation. A model with high capacity or many hyperparameters would confound representation effects with architecture-tuning effects. GCN's single design choice (layer count) makes the comparison clean.
2. **It matches the data scale.** 88 subjects and 19 nodes. GIN and deep ChebNet have far more capacity than 88 subjects can support. The brief explicitly warns against unnecessary complexity, and here complexity is not merely unnecessary — it is actively harmful to validity.
3. **It matches the graph structure.** With weighted, undirected, densely-connected 19-node connectivity graphs, first-order spectral convolution is exactly the right inductive bias. Only 2 layers are needed: a sparsified 19-node graph has a small diameter, so two hops of message passing already reach most of the scalp. **Deeper GCNs risk over-smoothing** — on a graph this small, additional layers drive node representations toward a common value. Layer depth is therefore tested explicitly as ablation 4 rather than assumed.
4. **It is explainable and reproducible** — a fixed, published, widely-implemented propagation rule.

### Candidates evaluated and rejected as *primary*

| Architecture | Verdict |
|---|---|
| **GAT** (`10.48550/arXiv.1710.10903`) | **Adopted as secondary sensitivity check only.** Learned attention re-weights edges, which would partially *undo* the edge definition we are trying to manipulate — it would confound Factor B. Valuable precisely as a robustness test, and [F7] explicitly calls for it. |
| **ChebNet** (`10.48550/arXiv.1606.09375`) | Adds hyperparameter K; K=1 reduces to GCN. Extra tuning surface with no benefit at this scale. |
| **GraphSAGE** (`10.48550/arXiv.1706.02216`) | Built for inductive sampling on large graphs; 19 nodes need no sampling. |
| **GIN** (`10.48550/arXiv.1810.00826`) | Maximum discriminative power (WL-test), but designed for graph isomorphism problems; severe overfitting risk at n=88. |
| **ST-GCN** | Would be the strongest *accuracy* choice by modelling window sequences, but adds a temporal dimension that confounds the representation comparison. **Recommended as a follow-up study, not the primary.** |

**If a reviewer demands a stronger model:** report GCN as the controlled primary and add GAT + ST-GCN as an appendix. Do not swap the primary — doing so would forfeit the paper's actual contribution.

---

# Section 11 — Graph Construction

**Recommended: Option 5 — Hybrid Graph.**

| Element | Recommendation |
|---|---|
| **Nodes** | 19 EEG channels (fixed 10–20 montage). Node identity is consistent across all subjects — a major advantage of this dataset. |
| **Node features** | **The independent variable** — pipeline-specific `X ∈ ℝ^(19×F)` (Section 9). |
| **Edges** | **Factor B, three levels:** (i) *spatial* — Euclidean distance between standard 10–20 electrode coordinates, Gaussian kernel; (ii) *functional* — wPLI (volume-conduction robust, per [F11]); (iii) *hybrid* — element-wise product of (i) and (ii). |
| **Edge weights** | Continuous, in [0,1]. Retain weights — do **not** binarise; [F9] explicitly recommends weighted graphs. |
| **Sparsification** | kNN with k=4, symmetrised via `A = max(A, Aᵀ)` (retains roughly a quarter of possible edges after symmetrisation). Report k ∈ {3,4,6} as a sensitivity analysis. |
| **Normalisation** | Symmetric: `Â = D̃^-½(A+I)D̃^-½` with self-loops. |
| **Direction** | Undirected (wPLI is symmetric). Granger/directed graphs deferred to future work. |
| **Graph per** | One graph **per window**; subject-level prediction by averaging window logits (Section 13). |

**Why hybrid over the alternatives**

- *Option 1 (electrode/spatial only):* identical for every subject and every class — it carries **zero discriminative information** on its own and can only act as a smoothing prior. Necessary as a control, insufficient as the answer.
- *Option 2 (functional only):* discriminative and neurophysiologically motivated, but volume conduction and reference choice inject spurious edges — mitigated but not eliminated by wPLI.
- *Option 3 (frequency-band graph):* nodes = bands gives only ~5 nodes; too small for meaningful message passing.
- *Option 4 (feature graph):* edges between features are statistically defined and lack neurophysiological grounding; hard to defend to a clinical reviewer.
- *Option 5 (hybrid):* keeps anatomically-real spatial structure while letting functional coupling carry subject-specific signal. It is also the configuration [R9] identifies as missing ("most studies overlook spatial connectivity between channels").

---

# Section 12 — Binary vs Multiclass

**Recommendation: 3-class multiclass (AD vs FTD vs CN) as primary, with binary AD vs CN reported as a secondary, comparability-only result.**

| Consideration | Assessment |
|---|---|
| Dataset availability | ds004504 provides all three labels in **one protocol** — no merging required |
| Biological validity | Strong: [F9] demonstrated AD and FTD produce *dissimilar* network disturbances |
| Number of subjects | 36 / 23 / 29 |
| Class balance | Moderately imbalanced (41% / 26% / 33%) — manageable with class weighting and macro-F1 |
| Label reliability | Clinical diagnosis + MMSE; CN group has MMSE 30.0±0.0 (a clean control group) |
| Clinical meaning | **High** — AD vs FTD differential diagnosis is a genuine and difficult clinical problem |
| Difficulty | Higher than binary, appropriately so |
| Existing SOTA | ≈80–85% under sound protocols per [R1] |
| Risk of confounding | **Low** — single site, single device, single protocol |
| Publication potential | Higher: 3-class is more informative and less saturated |

**Critical warning, per the brief's explicit instruction.** Multiclass is recommended here **only because all three classes come from the same acquisition protocol**. If a multiclass task required combining, say, ds004504 (AD/FTD, Greece, Nihon Kohden, 19 ch, 500 Hz, eyes-closed) with ds004584 (PD, Iowa, 63 ch, 500 Hz, eyes-open), **we would strongly advise against it**: the classifier would learn site, device, montage, and eyes-open/closed differences rather than pathology, and would report near-perfect, entirely spurious accuracy. **Do not build a "Healthy vs AD vs Parkinson's" model from different datasets.** This is the single most common fatal error in this literature.

Multiclass is *not* recommended merely because it is harder — it is recommended because the classes are scientifically comparable and protocol-matched.

---

# Section 13 — Experimental Protocol

### Design: two-factor, fully crossed

**Factor A — node-feature representation (6 levels):** P1 frequency · P2 connectivity-only · P3 time-frequency · P4 time-domain · P5 raw · P6 fused
**Factor B — edge definition (3 levels):** spatial · functional (wPLI) · hybrid

**6 × 3 = 18 cells × 5 seeds × 5 CV folds = 450 training runs.** Each run is ~3k parameters on ~7k windows — minutes on a single GPU. This is entirely feasible, and it is the design that isolates representation from architecture.

### Splitting — the most important methodological decision

- **Subject-level, stratified, nested 5-fold cross-validation.** All windows from a subject stay in exactly one fold. With 88 subjects: ~70 train / ~18 test per fold, with an inner split for validation.
- **All preprocessing that learns parameters is fit on the training fold only** — normalisation statistics, ICA where applicable, any feature scaling. Fitting a scaler on all 88 subjects before splitting is leakage.
- **Report subject-level predictions** (mean of window logits), not window-level accuracy. Window-level metrics inflate results because windows from one subject are highly correlated — this is exactly the failure [R1] identifies.
- **Fixed seeds** {0,1,2,3,4}; identical folds across all 18 cells so cells are paired.

### Controlled variables (held constant across all 18 cells)
dataset · subject splits and fold assignment · labels · GNN architecture and layer widths · optimiser (AdamW, lr 1e-3, weight decay 1e-4) · batch size 64 · max 200 epochs with early stopping (patience 20 on validation macro-F1) · class weighting (inverse frequency) · window length 10 s · sampling rate 128 Hz · random seeds.

### Metrics (imbalance-aware, per the brief)
**Primary: macro-F1.** Secondary: balanced accuracy, ROC-AUC (one-vs-rest, macro), PR-AUC, per-class sensitivity/specificity, Cohen's κ, and the 3×3 confusion matrix. Plain accuracy is reported but never used for ranking.

### Statistical testing
- Per-cell mean ± 95% CI over 5 folds × 5 seeds.
- **Two-way repeated-measures ANOVA** on macro-F1 with factors Representation and Edge-definition, plus their interaction — the interaction term is scientifically the most interesting quantity.
- Pairwise comparisons via **Wilcoxon signed-rank** on paired folds, **Holm–Bonferroni** corrected.
- Report **effect sizes** (Cohen's d / partial η²), not just p-values. With 88 subjects, insist on effect sizes.

### Ablations
1. k ∈ {3,4,6} for kNN sparsification.
2. Window length ∈ {5,10,20} s.
3. Edge weights binarised vs weighted.
4. GCN depth ∈ {1,2,3} (over-smoothing check).
5. GAT substituted for GCN (architecture sensitivity — addresses [F7]).
6. **Leakage demonstration:** deliberately re-run one cell with window-level random splitting to quantify how much accuracy inflates. This turns a methodological point into a measured result and is likely to be among the paper's most cited figures.

### Baselines (Section 17)
Non-graph baselines on identical splits: logistic regression and SVM on P1 features; random forest; EEGNet [F14]; a 1D-CNN. Plus the published GNN results of [F3]/[F4], compared honestly with a note on protocol differences.

### External validation
Repeat the *pipeline ranking* (not the model weights) on **ds004584** (PD vs control, 63 channels). The scientific question is whether the ordering of representations transfers across disease, montage, and site. This is a genuine generalisation test and materially strengthens the paper.

---

# Section 14 — Proposed Research Questions

**RQ1.** Under a fixed GCN and fixed subject-level splits, which EEG signal representation (raw, time-domain, frequency, time-frequency, connectivity, fused) yields the highest macro-F1 for 3-class AD/FTD/CN classification — and is the difference statistically significant after correction for multiple comparisons?

**RQ2.** Does functional-connectivity-derived graph structure (wPLI) outperform purely spatial electrode-distance graphs, and does a hybrid of the two outperform either alone?

**RQ3.** Do node-feature representation and edge definition **interact** — i.e. is the best representation different depending on how the graph is built? (Statistically the interaction term; scientifically the most novel question, and one no reviewed paper addresses.)

**RQ4.** How much of the reported performance spread in the ds004504 literature (≈80–99%) is attributable to representation choice, and how much to evaluation protocol? Quantified by re-running an identical pipeline under subject-level vs window-level splitting.

**RQ5.** Does the representation ranking established on AD/FTD transfer to an independent disease, montage, and site (Parkinson's, ds004584, 63 channels)?

*These improve on the brief's draft questions by making each one falsifiable, adding the interaction question (RQ3), and converting the vague "how sensitive is GNN performance to preprocessing" into a measurable variance decomposition (RQ4).*

---

# Section 15 — Proposed Paper Contributions

**Is the proposed contribution defensible?**
> "A controlled comparison of multiple EEG signal preprocessing and representation pipelines using a common Graph Neural Network architecture for neurological disorder classification."

**Yes — with one strengthening amendment.** As phrased it is a benchmarking paper, and benchmarking papers are vulnerable to "so what?" Adding the isolation claim converts it into a causal-inference claim:

> "The study isolates the effect of signal representation from model architecture by holding the downstream GNN constant."

This is defensible **and** verifiable here, because the dataset is single-protocol and the model is fixed by design. We further recommend adding the *interaction* claim (RQ3), which no reviewed paper makes.

### The four contributions

**1. Methodological.** A fully-crossed 6 × 3 factorial protocol that separates *signal representation* from *graph topology* from *model architecture* — three factors that the existing literature confounds. Includes a reusable, leakage-free, subject-level evaluation harness.

**2. Experimental.** The first systematic evidence on which EEG representation a GNN actually benefits from for dementia-subtype classification, with effect sizes and confidence intervals, plus a quantified measurement of how much window-level splitting inflates reported accuracy on this dataset.

**3. Scientific.** Evidence on whether AD and FTD are better distinguished by *node-local* spectral properties or by *inter-channel* coupling — a neurophysiologically meaningful question that speaks directly to the network-degeneration hypothesis of [F12] and the AD/FTD network dissimilarity of [F9].

**4. Reproducibility.** Open code, fixed seeds, fixed fold assignments published as files, and a CC0 dataset — meaning the entire study is re-runnable end-to-end by a third party. Directly responsive to the leakage critique in [R1] and [F8].

---

# Section 16 — Threats to Validity

| Threat | Severity | Mitigation |
|---|---|---|
| **Dataset size (88 subjects)** | **High** | Subject-level nested CV; 5 seeds; report CIs and effect sizes; never claim clinical readiness. Test folds are only ~18 subjects — state this plainly. |
| **Class imbalance (23 FTD)** | Medium | Macro-F1 as primary metric; class-weighted loss; stratified folds; per-class sensitivity reported. |
| **Subject leakage** | **High if mishandled** | Subject-level splitting enforced in code, with an assertion that train/test subject-ID sets are disjoint. |
| **Preprocessing leakage** | High | All fitted transforms (scalers, ICA) fit inside the training fold only. |
| **Overfitting to the benchmark** | Medium | Hyperparameters fixed a priori from literature, not tuned per cell; external validation on ds004584. |
| **Site/device homogeneity** | **Medium-High** | Single site and device means results may not generalise; explicitly scoped as a limitation, partially addressed by ds004584. |
| **Demographic confounding** | Medium | Groups differ in sex balance (AD 67% F vs CN 38% F) and slightly in age — **report a demographics-only baseline classifier** to show EEG adds information beyond age/sex. This is an important and easily-overlooked control. |
| **MMSE as a shortcut** | Medium | MMSE is near-perfectly separable (CN = 30.0±0.0). Never include MMSE as a model input; it is a label proxy, not a feature. |
| **Unequal recording lengths (307–1291 s)** | Medium | Subjects contribute unequal window counts — cap windows per subject, or weight subjects equally in the loss. Otherwise long recordings dominate. |
| **Graph construction assumptions** | Medium | k and edge definition treated as explicit experimental factors, not fixed guesses. |
| **Volume conduction inflating connectivity** | Medium | Use wPLI/PLI rather than raw coherence, per [F11]. |
| **Hyperparameter sensitivity** | Medium | Identical hyperparameters across all cells; sensitivity reported in ablations. |
| **Limited external validation** | Medium | ds004584 tests ranking transfer; BrainLat/CAUEEG proposed as future work. |
| **Negative-result risk** | Medium | If representation effects are small, that is a publishable finding given [R1]'s convergence observation — provided the study is adequately powered and honestly reported. |

---

# Section 17 — Benchmarking Against Existing Research (SOTA)

| Paper | Year | Dataset | Task | Model | Preprocessing | Metric | Reported result |
|---|---|---|---|---|---|---|---|
| [F2] DICE-Net | 2023 | ds004504 | AD vs CN | Dual-Input Convolution Encoder (Conv + Transformer Encoder + FF) | denoising → band power + coherence | Accuracy | **83.28%**, **Leave-One-Subject-Out** *(verified from abstract)* |
| [F2] DICE-Net | 2023 | ds004504 | FTD vs CN | as above | as above | — | **Not reported** in the abstract; full text paywalled |
| [R14] | 2025 | ds004504 | AD/FTD/CN | DL | resting-state EEG | Accuracy | Not verified (paywalled) |
| [R2] | 2026 | ds004504 | dementia | InceptionV3 | 5 TFRs compared | Accuracy | STFT best; exact value not extracted |
| [R10] Connectogram-COH | 2025 | ds004504 | AD | CNN on time-graphs | coherence dynamics | Accuracy | Not verified precisely |
| [F3] Klepl GNN | 2022 | private AD | AD vs CN | GNN | FC methods compared | Accuracy | Not verified (paywalled) |
| [R1] AHEPA review | 2026 | ds004504 | AD vs CN | *survey* | all | Accuracy | **≈80–85% under sound protocols** |
| [F10] | 2020 | private | dementia | multi-modal ML | multi-feature | Accuracy | Not verified (paywalled) |

**Baseline families to compare against (per brief §17):** Traditional ML (SVM/RF/LR on P1) · CNN (EEGNet [F14], 1D-CNN) · RNN/LSTM · Transformer (DICE-Net [F2]) · Hybrid ([F2]) · **existing GNN** ([F3], [F4], [R9], [R12]).

> **Honest caveat, and an important one.** Cross-paper accuracy comparison on this dataset is **not currently meaningful**, because studies differ in splitting (window vs subject), class pairing (binary vs 3-class), and window length. [R1] finds the field converges to ≈80–85% once protocols are sound, and that higher figures generally reflect leakage. **Our target is therefore not to beat 99% — it is to report trustworthy numbers in the 80–90% band with a protocol that others can reproduce.** Any submission claiming >95% on 88 subjects with subject-level splits should be treated as suspect, including our own.

---

# Section 18 — Research Gap Statement

Four concrete gaps, each traceable to quoted evidence.

**GAP 1 — Representation and architecture are confounded.**
Every reviewed study varies preprocessing *and* model together, so it is impossible to attribute performance differences to the signal representation rather than the classifier. [R2] is the closest counter-example — it compares five time-frequency representations under one CNN — but it uses no graph model, and its own Limitations section concedes it "does not explicitly model the true spatial topology or anatomical neighborhood relationships among electrodes."
*Evidence:* [R2] §4.4; [R1] Discussion.

**GAP 2 — Graph construction choices are asserted, not tested.**
GNN studies on EEG pick one edge definition and one sparsification rule, then report a single number. [F3] is the only reviewed work that systematically compares functional-connectivity methods as GNN inputs, and even it holds the node features fixed. Meanwhile [R9] states that "most studies overlook spatial connectivity between channels" and that "most existing research relies solely on functional connectivity methods … failing to fully capture the complex interactions."
*Evidence:* [R9] Introduction; [F3]; [F5] survey.

**GAP 3 — Reported performance is inflated by evaluation protocol, and the size of the inflation is unquantified.**
The literature on this dataset spans ≈80–99% accuracy. [R1] attributes the spread to "data leakage and overfitting to non-independent samples" rather than genuine representational advantage — but nobody has *measured* the inflation on a fixed pipeline.
*Evidence:* [R1] Discussion; [F8].

**GAP 4 — Explicit calls for graph modelling are unanswered in dementia subtyping.**
[R8] recommends verbatim that "future studies should consider employing novel techniques, such as graph convolutional networks … to explore the complex interactions between brain regions," and [F7] asks for systematic comparison across graph architectures. Neither has been carried out for AD/FTD/CN on a public dataset.
*Evidence:* [R8] Discussion; [F7] Discussion; [F9] Future directions.

### Which gap is most novel + feasible + testable?

**GAP 1, executed jointly with GAP 2** — i.e. the two-factor design. Assessment:
- *Novel:* no reviewed paper crosses representation with graph construction under a fixed model.
- *Feasible:* 450 runs of a 3k-parameter model on a 19.6-hour CC0 dataset — days of compute, not months.
- *Testable:* yields a signed, statistically-tested effect with confidence intervals, and a falsifiable interaction hypothesis.

GAP 3 is folded in as a single high-impact ablation (cheap, and it strengthens the paper's credibility). GAP 4 is the motivation, not a separate study.

---

# Section 19 — DOI Tracking (Mandatory)

### Machine-readable DOI list — final 30

```
R1  — 10.1007/s11571-026-10464-w
R2  — 10.3390/diagnostics16050746
R3  — 10.1038/s41598-026-42452-9
R4  — 10.1038/s41598-026-57069-1
R5  — 10.3390/brainsci16080856
R6  — 10.3390/brainsci16070716
R7  — 10.1016/j.mex.2026.103821
R8  — 10.1038/s41598-025-02018-7
R9  — 10.3389/fnins.2025.1555657
R10 — 10.3390/diagnostics15111441
R11 — 10.1142/s0129065725500480
R12 — 10.1007/s40747-025-01974-x
R13 — 10.1186/s12938-025-01361-0
R14 — 10.1007/s11571-025-10232-2
R15 — 10.1038/s41514-025-00243-y
R16 — 10.1016/j.compbiomed.2025.111041
F1  — 10.3390/data8060095
F2  — 10.1109/access.2023.3294618
F3  — 10.1109/tnsre.2022.3204913
F4  — 10.1109/tnsre.2023.3321634
F5  — 10.1109/tnsre.2024.3355750
F6  — 10.1002/hbm.25994
F7  — 10.1038/s41746-023-00983-9
F8  — 10.1016/j.patter.2023.100804
F9  — 10.1186/1471-2202-10-101
F10 — 10.1016/j.neunet.2019.12.006
F11 — 10.1002/hbm.20346
F12 — 10.1093/cercor/bhj127
F13 — 10.48550/arXiv.1609.02907
F14 — 10.1088/1741-2552/aace8c
```

The complete CSV (`Paper, Year, DOI, DOI Verified, Selected, Citations`) for **all 87 papers examined** is provided in the companion file **`doi_tracking.csv`**, and the full 1,091-record discovery corpus in **`papers_seen_full.csv`**.

**Papers with no DOI:** CHB-MIT origin (Shoeb 2009, MIT PhD thesis) — **DOI: Not available**. No DOI was invented anywhere in this report.

---

# Section 20 — Category Assignment (Brief §24)

Kept strictly separate, as instructed.

**Category A — Explicit GNN future work** (authors name GNN/graph neural/graph convolutional networks as future work; verbatim quotation extracted from retrieved full text):
- **[R8]** `10.1038/s41598-025-02018-7` — "Future studies should consider employing novel techniques, such as graph convolutional networks…"
- **[F7]** `10.1038/s41746-023-00983-9` — "Future studies should consider alternative methodologies (e.g. graph attention networks, graph neural ordinary differential equations) for graph learning."

**Category B — Graph/network/connectivity future work, without the term GNN:**
- **[R2]** `10.3390/diagnostics16050746` — "alternative graph-based or topology-aware fusion schemes may further enhance spatial interpretability and performance"
- **[F9]** `10.1186/1471-2202-10-101` — "Future directions" section; weighted-graph recommendation
- **[R13]**, **[R16]**, **[R5]**, **[F12]** — connectivity/network modelling advocated

**Category C — Strong GNN opportunity, no graph mention by authors:**
[R3], [R4], [R6], [R7], [R14], [R15], [F2], [F10], [F14]

**Category D — Already uses GNNs** (valuable for SOTA and graph-construction insight; **not counted toward the "GNN as future work" gap argument** unless independently qualifying):
[R9], [R10], [R11], [R12], [F3], [F4], [F5], [F6], [F7]*, [R1]*
*(\*[F7] appears in both A and D because it independently satisfies the Category A criterion; [R1] is a survey documenting the gap.)*

> **Discipline note.** No paper was placed in Category A on the basis of our own judgement that a GNN "could" be applied. Category A required a verbatim quotation from retrieved full text. For the 40 papers whose full text could not be retrieved (paywalled), no category-A claim is made — they are marked *"full text not retrievable"* rather than assigned by inference.

---

# Section 21 — Final Decision

```
RECOMMENDED DISEASE:
  Alzheimer's disease and frontotemporal dementia (dementia subtyping),
  with healthy controls. Weighted score 8.10/10 (Parkinson's 7.65 second).

RECOMMENDED DATASET:
  OpenNeuro ds004504 (AHEPA / University of Ioannina)
  DOI 10.18112/openneuro.ds004504.v1.0.8  [verified; v1.0.9 in the dataset's
  own metadata does NOT resolve]
  Dataset paper: 10.3390/data8060095
  88 subjects (36 AD / 23 FTD / 29 CN) · 19 ch · 500 Hz · eyes-closed ·
  19.61 h total · CC0
  External validation: OpenNeuro ds004584 (100 PD / 49 control, 63 ch)

RECOMMENDED TASK:
  Multiclass — 3-class AD vs FTD vs CN, subject-level.
  (Binary AD vs CN reported additionally for literature comparability.)
  Multiclass is safe here ONLY because all three classes share one protocol.

RECOMMENDED GNN:
  2-layer GCN (Kipf & Welling, 10.48550/arXiv.1609.02907), ~3k parameters,
  held constant across every experimental cell.
  GAT as a secondary sensitivity check only.

RECOMMENDED GRAPH:
  Hybrid. Nodes = 19 channels. Node features = pipeline-specific X ∈ R^(19×F).
  Edges = wPLI functional connectivity × spatial adjacency, kNN k=4, weighted,
  undirected, symmetric-normalised with self-loops.

RECOMMENDED PIPELINES:
  1. P1  Frequency-domain band power + spectral entropy      (X ∈ R^(19×7))
  2. P2  Connectivity-only (wPLI/PLV/coherence)              (X ∈ R^(19×5))
  3. P3  Time-frequency (CWT/DWT sub-band energy+entropy)    (X ∈ R^(19×20))
  4. P4  Time-domain statistical + Hjorth + entropy          (X ∈ R^(19×10))
  5. P5  Raw/minimal (z-scored windows)                      (X ∈ R^(19×1280))
  6. P6  Fused (P1 + P3 + graph metrics)                     (X ∈ R^(19×~34))

PRIMARY RESEARCH GAP:
  Existing EEG dementia-classification studies vary signal representation and
  model architecture simultaneously, and evaluate with protocols that
  frequently leak across subjects. Consequently it is unknown how much of
  reported performance comes from the signal representation, how much from
  the graph construction, and how much from evaluation protocol. No study
  crosses representation with graph construction under a fixed GNN.

PROPOSED NOVELTY:
  A fully-crossed 6 (representation) x 3 (edge definition) factorial study
  under one fixed GCN, with subject-level nested CV, seeded and reproducible,
  reporting effect sizes and the representation x topology interaction —
  plus a direct measurement of how much window-level splitting inflates
  accuracy on this dataset.

EXPECTED BASELINES:
  LR / SVM / RF on band power; EEGNet (10.1088/1741-2552/aace8c); 1D-CNN;
  DICE-Net (10.1109/access.2023.3294618, 83.28% AD/CN, LOSO validation);
  published GNN results (10.1109/tnsre.2022.3204913, 10.1109/tnsre.2023.3321634).

BIGGEST RISK:
  88 subjects. Subject-level test folds contain only ~18 subjects, so
  confidence intervals will be wide and small representation differences may
  not reach significance. Mitigated by 5x5 repeated CV, paired statistics,
  effect-size reporting, and external validation on ds004584 — and by the
  fact that a well-powered null result is publishable in light of the
  AHEPA benchmark's convergence finding.

GO / NO-GO:
  GO.
  The dataset is open, verified, and single-protocol; the gap is documented
  by independent 2025-2026 sources including two verbatim calls for graph
  methods; the compute is trivial; and the design is falsifiable either way.
  The one condition: commit to subject-level evaluation and publish the fold
  assignments. Without that, the study is worthless regardless of results.
```

### "If you were conducting this research yourself, exactly what experiment would you run?"

I would run this, in this order:

**Week 1–2 — Build the harness before any modelling.** Download ds004504 (CC0, direct S3, no application). Write one preprocessing module with the fixed preamble of Section 9 and a `Representation` interface with six implementations. Write the splitter first and unit-test it: assert that `set(train_subject_ids) & set(test_subject_ids) == ∅`, and freeze fold assignments to a JSON file committed to the repo. **Do not train anything until this test passes.**

**Week 3 — Establish the floor and the ceiling.** Run a demographics-only classifier (age + sex → 3 classes). This is the number that everything must beat; if band power cannot beat age and sex, the study has a confound, not a finding. Then run logistic regression on P1 band power for a classical reference point.

**Week 4–5 — The main grid.** All 18 cells × 5 folds × 5 seeds = 450 runs, identical hyperparameters throughout. Log everything: per-fold macro-F1, confusion matrices, training curves, wall-clock, and the exact commit hash.

**Week 6 — The ablation that makes the paper.** Take the single best cell and re-run it with window-level random splitting instead of subject-level. Report both numbers side by side. My expectation, based on [R1], is that accuracy jumps by 10–20 points purely from the split change. That single figure — same pipeline, same model, same data, two splitting protocols — is the most valuable thing this project can produce, because it gives the field a calibrated correction factor for interpreting the existing ds004504 literature.

**Week 7 — Transfer.** Re-run the representation ranking on ds004584 (PD, 63 channels). Do not expect the accuracies to transfer; ask only whether the *ordering* does.

**Week 8 — Statistics and writing.** Two-way RM-ANOVA, Holm-corrected Wilcoxon pairs, effect sizes. Write the limitations section honestly, especially the 18-subject test fold.

**What I would refuse to do:** tune hyperparameters per cell (it destroys the controlled comparison); report window-level accuracy as a headline; add architectural complexity to chase a higher number; or merge datasets to manufacture a multiclass problem.

**What would make me abandon the project:** if the demographics-only baseline reached ~80% macro-F1, the dataset's groups would be demographically confounded and no EEG result would be interpretable. This is a one-day check and it should be done first.

---

# Section 22 — Evidence Chain

```
Existing neurological classification research
   ↓  [F10] [F2] [R14] — dementia classified via ML/CNN/Transformer
Different signal-processing / preprocessing approaches
   ↓  [R2] 5 TFRs · [R15] complexity · [R3] spectral+FC · [R16] Morlet+MI
Limitations / inconsistencies in signal representation
   ↓  [R1] "≈80-85%… these extensions do not demonstrate a systematic
   ↓       performance improvement" + leakage finding
Insufficient modeling of relationships between channels / regions
   ↓  [R2] "does not explicitly model the true spatial topology…"
   ↓  [F6] "neglecting the functional connectivity features"
   ↓  [R9] "most studies overlook spatial connectivity between channels"
Opportunity for graph representation
   ↓  [R8] "Future studies should consider… graph convolutional networks"  (Cat A)
   ↓  [F7] "Future studies should consider alternative methodologies…"      (Cat A)
   ↓  [F9] "Future directions: Graph theory offers a growing amount of
   ↓       techniques to describe topological network features"             (Cat B)
GNN as a unified downstream classifier
   ↓  [F13] GCN · [F5] GNN-EEG survey · [F3] [F4] existing AD GNNs
Controlled preprocessing comparison
   ↓  6 representations × 3 edge definitions, one fixed GCN,
   ↓  subject-level nested CV, 450 seeded runs
Novel and reproducible experiment
   ↓
dataset selection → preprocessing implementation → graph construction →
GNN training → baseline comparison → ablation → statistical evaluation →
research paper
```

---

## Appendix A — Reproducibility of this review

| Artefact | File |
|---|---|
| Full discovery corpus (1,091 records) | `papers_seen_full.csv` |
| DOI tracking CSV (87 examined) | `doi_tracking.csv` |
| Verified metadata + tri-source citations | `verified_metadata.json` |
| Extracted full-text evidence quotations | `gnn_evidence.json` |

**Method:** OpenAlex API (discovery, 37 queries) → Crossref/DataCite (DOI verification) → Semantic Scholar (third citation source) → Europe PMC / arXiv / publisher OA (full-text retrieval, 39 of 87 succeeded) → manual reading for category assignment.

**Stated limitations of this review:**
1. Full text could not be retrieved for 48 of 87 examined papers (paywalls). For those, no internal-content claim is made and no Category A assignment was possible.
2. Citation counts differ between sources; all three are reported rather than one. One record (`10.1088/1741-2552/ab260c`) shows OpenAlex 76 vs Crossref 1,331 vs S2 1,400, indicating a split record in OpenAlex — a concrete illustration of why the brief's instruction not to rely on a single source matters.
3. 2026 papers have near-zero citations by construction and were judged on venue and content.
4. Discovery used OpenAlex as the primary index; a supplementary IEEE Xplore and PubMed sweep would likely surface additional paywalled engineering papers.
