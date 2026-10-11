"""Could-fix analyses: severity stratification (MMSE, analysis only) and the EEGNet dropout sensitivity."""
import numpy as np
import pandas as pd
import pytest

from eegrep.stats_could import dropout_sensitivity, severity, stratify, subject_predictions

NAMES = ["AD", "FTD", "CN"]


def _participants(rng, n=(12, 10, 10)):
    rows = []
    for c, k in enumerate(n):
        for i in range(k):
            mmse = 30.0 if c == 2 else float(rng.integers(8, 29))
            rows.append({"subject": f"s{c}-{i}", "group": "AFC"[c], "label": c, "age": 65.0, "sex_male": i % 2,
                         "mmse": mmse})
    return pd.DataFrame(rows)


def _runs(rng, part, pipelines=("P1", "P3"), edges=("spatial",), mild_bias=0.0):
    """Five repeats x five folds; every subject tested once per repeat. Milder patients (higher MMSE) get a lower
    true-class probability (shifted towards CN) by mild_bias per MMSE point above 20."""
    subs = part.subject.tolist()
    y = dict(zip(part.subject, part.label)); mm = dict(zip(part.subject, part.mmse))
    rows = []
    for p in pipelines:
        for e in edges:
            for seed in range(5):
                order = rng.permutation(subs)
                for fold, chunk in enumerate(np.array_split(order, 5)):
                    probs = []
                    for s in chunk:
                        pt = 0.6 - (mild_bias * max(mm[s] - 20, 0) if y[s] != 2 else 0) + rng.normal(0, 0.05)
                        pt = float(np.clip(pt, 0.05, 0.9))
                        pr = np.full(3, (1 - pt) / 2); pr[y[s]] = pt
                        if y[s] != 2 and pt < 0.4:                    # errors of mild patients go to CN
                            pr[:] = 0.1; pr[y[s]] = pt; pr[2] = 0.9 - pt
                        probs.append(pr.tolist())
                    rows.append({"tag": "main", "pipeline": p, "edge": e, "seed": seed, "fold": fold,
                                 "unit": f"{seed}-{fold}", "macro_f1": rng.normal(0.55, 0.05),
                                 "balanced_acc": 0.5, "roc_auc": 0.7,
                                 "preds": {"subjects": list(chunk), "y": [int(y[s]) for s in chunk], "prob": probs}})
    return pd.DataFrame(rows)


def test_subject_predictions_and_strata(rng):
    part = _participants(rng)
    long = subject_predictions(_runs(rng, part))
    assert len(long) == 2 * 5 * len(part) and set(long.pred) <= {0, 1, 2}
    per = long.groupby(["subject", "true_class"], as_index=False)["p_true"].mean()
    st = stratify(per, part)
    assert (st.loc[st.mmse >= 21, "stratum"] == "mild").all()
    assert (st.loc[st.mmse <= 20, "stratum"] == "moderate-severe").all()


def test_severity_detects_harder_mild_patients(rng):
    part = _participants(rng, (30, 30, 10))
    runs = _runs(rng, part, mild_bias=0.04)
    subj, summ, tests, reps = severity({"selection": runs}, part, NAMES)
    s = summ.set_index(["group", "stratum"])
    assert s.loc[("AD", "mild"), "p_true"] < s.loc[("AD", "moderate-severe"), "p_true"]
    assert s.loc[("AD", "mild"), "pred_cn"] > s.loc[("AD", "moderate-severe"), "pred_cn"]
    sp = tests[(tests.group == "AD") & tests.test.str.startswith("Spearman")].iloc[0]
    assert sp.effect < 0 and sp.p_holm < 0.05
    assert "CN" not in set(summ.group) and set(reps.pipeline) == {"P1", "P3"}
    assert (tests.p_holm >= tests.p - 1e-12).all()


def test_severity_null_has_no_effect(rng):
    part = _participants(rng, (30, 30, 10))
    _, _, tests, _ = severity({"selection": _runs(rng, part, mild_bias=0.0)}, part, NAMES)
    assert (tests.p_holm > 0.01).all()


def test_dropout_sensitivity_pairs_and_holm(rng):
    part = _participants(rng)
    main = _runs(rng, part, pipelines=("P3", "P5"), edges=("spatial", "hybrid"))
    main.loc[(main.pipeline == "P3") & (main.edge == "spatial"), "macro_f1"] += 0.1   # P3 x spatial best, as in data
    old = main[(main.pipeline == "P5") & (main.edge == "spatial")].assign(tag="deep_eegnet", edge="identity")
    new = old.assign(tag="deep_eegnet_drop025", macro_f1=old.macro_f1 + 0.02)
    tab = dropout_sensitivity(main, old, new)
    assert len(tab) == 3 and tab.loc[tab.reference == "EEGNet (dropout 0.5)", "delta"].iloc[0] == pytest.approx(0.02)
    assert (tab.nb_p_holm >= tab.nb_p - 1e-12).all()
