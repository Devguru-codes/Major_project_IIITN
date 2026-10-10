"""Publication figures (IMPLEMENTATION_PLAN.md §7). Every figure is built from
saved results files, never from hand-typed numbers; each is written as vector PDF
and 300-dpi PNG with the Okabe–Ito colour-blind-safe palette.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GROUP_COLORS = {"AD": "#D55E00", "FTD": "#0072B2", "CN": "#009E73"}
SEQ = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 100})


def save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def _info(cfg):
    import mne

    ds = cfg["dataset"]
    info = mne.create_info([ds["channel_rename"].get(c, c) for c in ds["channels"]], 128.0, "eeg")
    info.set_montage("standard_1020")
    return info


def _topo(ax, values, info, vlim=None, cmap="viridis"):
    import mne

    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=vlim or (None, None),
                                 contours=0, sensors=True)
    return im


def _groupwise(ax, data: dict, ylabel: str, points=True):
    """Box + jittered points per group (raincloud-style without the cloud)."""
    names = list(data)
    ax.boxplot([data[n] for n in names], widths=0.5, showfliers=False,
               medianprops={"color": "black"}, tick_labels=names)
    if points:
        rng = np.random.default_rng(0)
        for i, n in enumerate(names, 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(data[n])), data[n], s=10, alpha=0.7,
                       color=GROUP_COLORS.get(n, SEQ[(i - 1) % len(SEQ)]), zorder=3)
    ax.set_ylabel(ylabel)


# ---------- data / PSWE figures ----------

def fig_cohort(subj: pd.DataFrame, out_dir):
    """Fig 2: age, sex and recording length per group. `subj` = pswe_subject_table.csv."""
    order = ["AD", "FTD", "CN"]
    g = subj["group"].map({"A": "AD", "F": "FTD", "C": "CN"}).fillna(subj["group"])
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
    _groupwise(ax[0], {k: subj.loc[g == k, "age"].values for k in order}, "Age (years)")
    male = [subj.loc[g == k, "sex_male"].mean() for k in order]
    ax[1].bar(order, [1 - m for m in male], color=[GROUP_COLORS[k] for k in order], label="female")
    ax[1].bar(order, male, bottom=[1 - m for m in male], color="lightgrey", label="male")
    ax[1].set_ylabel("Proportion"); ax[1].legend(frameon=False, fontsize=7)
    _groupwise(ax[2], {k: subj.loc[g == k, "duration_s"].values / 60 for k in order}, "Recording (min)")
    save(fig, out_dir, "fig02_cohort")


def fig_band_topomaps(cache, cfg, out_dir):
    """Fig 3: relative band power per group as scalp topomaps (rows = groups, cols = bands)."""
    info = _info(cfg)
    names = cfg["dataset"]["class_names"]
    bands = list(cfg["bands"])
    p1 = cache.X("P1")
    subj_mean = pd.DataFrame(p1[..., :5].reshape(len(p1), -1)).groupby(cache.subject).mean()
    lab = pd.Series(cache.label).groupby(cache.subject).first()
    fig, ax = plt.subplots(3, 5, figsize=(9, 5.6))
    for j, b in enumerate(bands):
        vals = [subj_mean[lab == c].mean().values.reshape(19, 5)[:, j] for c in range(3)]
        vlim = (min(v.min() for v in vals), max(v.max() for v in vals))
        for c in range(3):
            im = _topo(ax[c, j], vals[c], info, vlim)
            if c == 0:
                ax[c, j].set_title(b)
            if j == 0:
                ax[c, j].set_ylabel(names[c])
        fig.colorbar(im, ax=ax[:, j], shrink=0.6, orientation="horizontal", pad=0.02)
    save(fig, out_dir, "fig03_band_topomaps")


def fig_pswe_example(data, sfreq, cfg, channels, channel, out_dir, t_span=None):
    """Fig 4: EEG trace with the MPF series and detected PSWEs shaded."""
    from .pswe import detect

    events, mpf, mask, step = detect(data, sfreq, cfg["pswe"], channels)
    c = channels.index(channel)
    t0, t1 = t_span or (0, min(120, data.shape[1] / sfreq))
    t = np.arange(data.shape[1]) / sfreq
    m = (t >= t0) & (t < t1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True, height_ratios=[2, 1])
    ax[0].plot(t[m], data[c, m], lw=0.5, color="black"); ax[0].set_ylabel(f"{channel} (µV)")
    ts = np.arange(mpf.shape[1]) * step + cfg["pswe"]["win_s"] / 2
    ms = (ts >= t0) & (ts < t1)
    ax[1].plot(ts[ms], mpf[c, ms], color=SEQ[4]); ax[1].axhline(cfg["pswe"]["mpf_threshold_hz"], ls="--", color="grey")
    ax[1].set_ylabel("MPF (Hz)"); ax[1].set_xlabel("Time (s)")
    for _, e in events[events.channel == channel].iterrows():
        if e.onset_s < t1 and e.onset_s + e.duration_s > t0:
            for a in ax:
                a.axvspan(max(t0, e.onset_s), min(t1, e.onset_s + e.duration_s), color=SEQ[5], alpha=0.2)
    save(fig, out_dir, "fig04_pswe_example")


def fig_pswe_burden(subj: pd.DataFrame, topo: pd.DataFrame, cfg, out_dir):
    """Fig 5: PSWE rate per group, per-channel topography, PSWE rate vs MMSE in patients."""
    from scipy import stats

    order = ["AD", "FTD", "CN"]
    g = subj["label"].map(dict(enumerate(cfg["dataset"]["class_names"])))
    info = _info(cfg)
    fig = plt.figure(figsize=(11, 3.2), layout="constrained")
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1, 1, 1, 1.6])
    ax0 = fig.add_subplot(gs[0])
    _groupwise(ax0, {k: subj.loc[g == k, "rate_per_min"].values for k in order}, "PSWE rate (events/min)")
    names = [cfg["dataset"]["channel_rename"].get(c, c) for c in cfg["dataset"]["channels"]]
    tp = topo.rename(index=cfg["dataset"]["channel_rename"]).reindex(names)
    vlim = (0, float(np.nanmax(tp[order].values)))
    topo_axes = []
    for i, k in enumerate(order):
        a = fig.add_subplot(gs[1 + i]); im = _topo(a, tp[k].values, info, vlim, cmap="magma"); a.set_title(k)
        topo_axes.append(a)
    fig.colorbar(im, ax=topo_axes, orientation="horizontal", shrink=0.6, label="median PSWE rate (events/min)")
    ax5 = fig.add_subplot(gs[4])
    pat = subj[g != "CN"]
    for k in ("AD", "FTD"):
        s = pat[g[g != "CN"] == k]
        ax5.scatter(s["mmse"], s["rate_per_min"], s=12, color=GROUP_COLORS[k], label=k)
    rho = stats.spearmanr(pat["mmse"], pat["rate_per_min"])
    ax5.set_xlabel("MMSE (analysis only)"); ax5.set_ylabel("PSWE rate")
    ax5.set_title(f"Spearman ρ={rho.statistic:.2f}, p={rho.pvalue:.3f}", fontsize=8)
    ax5.legend(frameon=False, fontsize=7)
    save(fig, out_dir, "fig05_pswe_burden")


# ---------- grid figures ----------

def fig_main_effects(df: pd.DataFrame, out_dir):
    """Fig 6 + 7: macro-F1 per representation and per edge type (each point = one test split)."""
    for factor, name in (("pipeline", "fig06_representation"), ("edge", "fig07_edge")):
        wide = df.groupby(["unit", factor])["macro_f1"].mean().unstack()
        wide = wide[wide.mean().sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(1 + 0.8 * wide.shape[1], 3))
        _groupwise(ax, {c: wide[c].values for c in wide.columns}, "Macro-F1 (subject level)")
        for i, c in enumerate(wide.columns, 1):
            m, se = wide[c].mean(), wide[c].std() / np.sqrt(len(wide))
            ax.errorbar(i + 0.3, m, yerr=1.96 * se, fmt="o", color="black", ms=3, capsize=2)
        ax.axhline(1 / 3, ls=":", color="grey")
        save(fig, out_dir, name)


def fig_interaction(cells: pd.DataFrame, out_dir):
    """Fig 8: 7×3 heatmap of mean macro-F1 + interaction line plot."""
    piv = cells.pivot(index="pipeline", columns="edge", values="macro_f1_mean")
    piv = piv[[c for c in ["spatial", "functional", "hybrid"] if c in piv.columns]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), width_ratios=[1, 1.3], layout="constrained")
    im = ax[0].imshow(piv.values, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(piv.shape[1]), piv.columns); ax[0].set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if piv.values[i, j] < np.nanmean(piv.values) else "black")
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="macro-F1")
    for k, (p, row) in enumerate(piv.iterrows()):
        ax[1].plot(piv.columns, row.values, marker="o", color=SEQ[k % len(SEQ)], label=p)
    ax[1].set_ylabel("macro-F1"); ax[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, out_dir, "fig08_interaction")


def fig_cd(cd: dict, out_dir):
    """Fig 9: critical-difference diagram (average rank, lower = better; bars join cells within CD)."""
    ranks = pd.Series(cd["avg_rank"]).sort_values()
    k = len(ranks)
    vals = ranks.values
    groups = []                                  # maximal runs of cells within one CD of each other
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd["cd"]:
            j += 1
        if j > i and (not groups or j > groups[-1][1]):
            groups.append((i, j))
    lo, hi = vals.min() - 0.5, max(vals.max() + 0.5, vals.min() + cd["cd"] + 1)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.2 * (k + len(groups))), layout="constrained")
    ax.set_xlim(hi, lo); ax.set_ylim(-len(groups) - 0.5, k + 1.5)
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    for i, (name, r) in enumerate(ranks.items()):
        y = k - i
        ax.plot([r, r], [y, k + 0.6], color="grey", lw=0.5)
        ax.text(r, y, f" {name.replace('×', ' × ')} ({r:.1f})", fontsize=6.5, va="center")
    for n, (i, j) in enumerate(groups):
        ax.plot([vals[i], vals[j]], [-0.6 - n] * 2, color="black", lw=2.5, solid_capstyle="butt")
    x0 = lo + 0.2
    ax.plot([x0, x0 + cd["cd"]], [k + 1.1] * 2, color=SEQ[5], lw=2)
    ax.text(x0 + cd["cd"], k + 1.1, f"  CD = {cd['cd']:.2f}", fontsize=7, va="center", ha="right")
    ax.set_xlabel(f"Average rank across 25 splits (lower = better; Friedman p = {cd['friedman_p']:.2g}); "
                  "black bars join cells not separated by Nemenyi", fontsize=8)
    save(fig, out_dir, "fig09_critical_difference")


def fig_confusions(df: pd.DataFrame, class_names, out_dir):
    """Fig 10: row-normalised confusion matrices (summed over 25 runs) of the best cell per representation."""
    best = df.groupby(["pipeline", "edge"])["macro_f1"].mean().reset_index().sort_values("macro_f1", ascending=False)
    best = best.drop_duplicates("pipeline").sort_values("pipeline")
    n = len(best)
    fig, ax = plt.subplots(1, n, figsize=(2.1 * n, 2.4))
    for a, (_, b) in zip(np.atleast_1d(ax), best.iterrows()):
        cm = np.sum([np.array(c) for c in df[(df.pipeline == b.pipeline) & (df.edge == b.edge)]["confusion"]], axis=0)
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if cmn[i, j] > 0.5 else "black")
        a.set_xticks(range(3), class_names, fontsize=7); a.set_yticks(range(3), class_names, fontsize=7)
        a.set_title(f"{b.pipeline}×{b.edge}\nF1={b.macro_f1:.2f}", fontsize=7)
    np.atleast_1d(ax)[0].set_ylabel("true"); fig.supxlabel("predicted", fontsize=8)
    save(fig, out_dir, "fig10_confusion_matrices")


def fig_roc_pr(df: pd.DataFrame, cells: list[tuple[str, str]], class_names, out_dir):
    """Fig 11: one-vs-rest ROC and PR per class, mean ± SD over repeats, for selected cells."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(2, 3, figsize=(9, 5.4), sharex=True, sharey=True)
    for k, (p, e) in enumerate(cells):
        cell = df[(df.pipeline == p) & (df.edge == e)]
        for c, cname in enumerate(class_names):
            rocs, prs = [], []
            for _, g in cell.groupby("seed"):
                y = np.concatenate([q["y"] for q in g["preds"]]) == c
                s = np.concatenate([q["prob"] for q in g["preds"]])[:, c]
                fpr, tpr, _ = roc_curve(y, s); rocs.append(np.interp(grid, fpr, tpr))
                pr, rc, _ = precision_recall_curve(y, s); prs.append(np.interp(grid, rc[::-1], pr[::-1]))
            for row, curves in ((0, rocs), (1, prs)):
                m, sd = np.mean(curves, 0), np.std(curves, 0)
                ax[row, c].plot(grid, m, color=SEQ[k % len(SEQ)], label=f"{p}×{e}")
                ax[row, c].fill_between(grid, m - sd, m + sd, color=SEQ[k % len(SEQ)], alpha=0.15)
            ax[0, c].set_title(f"{cname} vs rest")
    for c in range(3):
        ax[0, c].plot([0, 1], [0, 1], ls=":", color="grey"); ax[1, c].set_xlabel("FPR / recall")
    ax[0, 0].set_ylabel("TPR"); ax[1, 0].set_ylabel("precision"); ax[0, 0].legend(frameon=False, fontsize=7)
    save(fig, out_dir, "fig11_roc_pr")


def fig_seed_variance(df: pd.DataFrame, out_dir):
    """Fig S3: per-repeat mean macro-F1 for every cell (seed sensitivity)."""
    per = df.groupby(["pipeline", "edge", "seed"])["macro_f1"].mean().reset_index()
    per["cell"] = per["pipeline"] + "×" + per["edge"]
    order = per.groupby("cell")["macro_f1"].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, c in enumerate(order):
        v = per[per.cell == c]
        ax.scatter([i] * len(v), v["macro_f1"], c=[SEQ[s % len(SEQ)] for s in v["seed"]], s=14)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7); ax.set_ylabel("macro-F1 (repeat mean)")
    save(fig, out_dir, "figS3_seed_variance")


# ---------- ablation figures ----------

def fig_leakage(leak: pd.DataFrame, out_dir):
    """Fig 12: identical pipeline/model/data, only the split protocol differs."""
    cells = list(leak["cell"].unique())
    fig, ax = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 3), sharey=True, squeeze=False)
    for a, cell in zip(ax[0], cells):
        g = leak[leak.cell == cell].set_index("protocol")
        x = np.arange(2)
        for j, (col, lab) in enumerate((("subject_macro_f1", "subject-level metric"),
                                        ("window_macro_f1", "window-level metric"))):
            a.bar(x + (j - 0.5) * 0.38, g[col].values, 0.38, yerr=g[f"{col}_sd"].values, capsize=3,
                  color=SEQ[4] if j == 0 else SEQ[1], label=lab)
        a.set_xticks(x, [p.replace(" (", "\n(") for p in g.index], fontsize=7)
        a.set_title(cell.replace("x", " × "), fontsize=8)
        a.axhline(1 / 3, ls=":", color="grey")
        a.set_ylim(0, 1.05)
    ax[0, 0].set_ylabel("Macro-F1"); ax[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, out_dir, "fig12_leakage")


def fig_confirmation(main_cells: pd.DataFrame, confirm_cells: pd.DataFrame, out_dir):
    """Fig S6: every cell's macro-F1 on the original splits (seeds 0–4) vs the independent confirmation
    splits (seeds 5–9). Points below the identity line lost performance on fresh splits (selection bias)."""
    m = main_cells.set_index(["pipeline", "edge"])["macro_f1_mean"]
    c = confirm_cells.set_index(["pipeline", "edge"])["macro_f1_mean"].reindex(m.index)
    rho = m.corr(c, method="spearman")
    fig, ax = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    lo, hi = min(m.min(), c.min()) - 0.02, max(m.max(), c.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls=":", color="grey")
    reps = sorted({p for p, _ in m.index})
    for (p, e), x in m.items():
        ax.scatter(x, c[(p, e)], color=SEQ[reps.index(p) % len(SEQ)], s=22,
                   marker={"spatial": "o", "functional": "s", "hybrid": "^"}.get(e, "o"))
    for k, p in enumerate(reps):
        ax.scatter([], [], color=SEQ[k % len(SEQ)], label=p)
    ax.legend(frameon=False, fontsize=7, ncol=2, title="o spatial  □ functional  △ hybrid", title_fontsize=6)
    ax.set_xlabel("Macro-F1, seeds 0–4 (selection)"); ax.set_ylabel("Macro-F1, seeds 5–9 (confirmation)")
    ax.set_title(f"Spearman ρ = {rho:.2f} across 21 cells", fontsize=8)
    save(fig, out_dir, "figS6_confirmation")


def fig_es_sensitivity(reps: pd.DataFrame, runs: dict, out_dir):
    """Fig S7: macro-F1 per representation (mean and 95% CI over the 25 splits, averaged over graph types)
    under each stopping rule (left), and the epoch at which the kept weights were taken (right)."""
    rules = list(dict.fromkeys(reps["rule"]))
    pipes = sorted(reps["pipeline"].unique())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), width_ratios=[2.2, 1], layout="constrained")
    for k, rule in enumerate(rules):
        g = reps[reps.rule == rule].set_index("pipeline").reindex(pipes)
        x = np.arange(len(pipes)) + (k - (len(rules) - 1) / 2) * 0.22
        ax.errorbar(x, g["macro_f1"], yerr=[g["macro_f1"] - g["ci_lo"], g["ci_hi"] - g["macro_f1"]], fmt="o",
                    ms=4, capsize=2, color=SEQ[k % len(SEQ)], label=rule)
    ax.axhline(1 / 3, ls=":", color="grey", lw=0.8)
    ax.set_xticks(range(len(pipes)), pipes); ax.set_ylabel("Subject-level macro-F1")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    data = [runs[r]["best_epoch"].dropna().to_numpy() for r in rules]
    bx.boxplot(data, showfliers=False, widths=0.5)
    bx.set_xticks(range(1, len(rules) + 1), [r.split(" (")[0] for r in rules], fontsize=7, rotation=15)
    bx.set_ylabel("Epoch of kept weights")
    save(fig, out_dir, "figS7_early_stopping")


def fig_forest(abl: pd.DataFrame, out_dir, name: str = "fig13_ablation_forest"):
    """Fig 13: Δ macro-F1 (ablated − reference) with 95% CI, per ablation and cell; * = Holm NB-corrected p < .05."""
    cells = list(abl["cell"].unique())
    names = sorted(abl["ablation"].unique())
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(names) + 1))
    for k, cell in enumerate(cells):
        g = abl[abl.cell == cell].set_index("ablation").reindex(names)
        y = np.arange(len(names)) + (k - (len(cells) - 1) / 2) * 0.25
        ax.errorbar(g["delta"], y, xerr=[g["delta"] - g["delta_ci_lo"], g["delta_ci_hi"] - g["delta"]],
                    fmt="o", ms=4, capsize=2, color=SEQ[[4, 5][k % 2]], label=cell.replace("x", " × "))
        for yi, (_, r) in zip(y, g.iterrows()):
            if r["nb_p_holm"] < 0.05:
                ax.text(r["delta_ci_hi"] + 0.004, yi, "*", va="center", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Δ macro-F1 vs reference cell (paired, 25 splits)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out_dir, name)


def fig_permutation(perm: dict, out_dir):
    """Fig S5: label-permutation null distribution vs observed macro-F1."""
    fig, ax = plt.subplots(1, len(perm), figsize=(4 * len(perm), 2.8), squeeze=False)
    for a, (cell, r) in zip(ax[0], perm.items()):
        a.hist(r["null"], bins=25, color="lightgrey", edgecolor="grey")
        a.axvline(r["observed_seed0_macro_f1"], color=SEQ[5], lw=2)
        a.set_title(f"{cell.replace('x', ' × ')}: p = {r['p']:.3f} (n = {r['n_perm']})", fontsize=8)
        a.set_xlabel("Macro-F1 (5-fold mean)")
    save(fig, out_dir, "figS5_permutation_null")
