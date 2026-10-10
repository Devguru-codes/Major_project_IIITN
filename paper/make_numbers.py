"""Write paper/generated/numbers.tex: LaTeX macros for every number quoted in the text, read from the
committed result files (reports/), so the prose never contains a hand-typed result.

Usage: python paper/make_numbers.py   (after copying NB07 outputs into reports/nb07/)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT / "paper" / "generated" / "numbers.tex"
AD, FTD = "C(group, Treatment('CN'))[T.AD]", "C(group, Treatment('CN'))[T.FTD]"
macros: dict[str, str] = {}


def name(s: str) -> str:
    """LaTeX macro names are letters only."""
    digits = "zero one two three four five six seven eight nine".split()
    s = "".join(digits[int(c)].capitalize() if c.isdigit() else c for c in s)
    return re.sub(r"[^A-Za-z]", "", s)


def put(key: str, value, fmt: str = "{:.3f}") -> None:
    s = fmt.format(value) if not isinstance(value, str) else value
    if s.startswith("-"):                       # typographic minus, valid in text and math mode
        s = r"\ensuremath{-}" + s[1:]
    macros[name(key)] = s


def p_fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def p_txt_(p: float) -> str:
    return "$p<$0.001" if p < 0.001 else f"$p=${p:.3f}"


# ---- main grid (NB07) ----
nb07 = REP / "nb07"
summ = json.loads((nb07 / "summary.json").read_text())
cells = pd.read_csv(nb07 / "tables" / "grid_cells.csv")
put("nMainRuns", summ["n_main_runs"], "{:,}")
put("nAblRuns", summ["n_ablation_runs"], "{:,}")
put("cpuHoursGrid", summ["cpu_hours_main_grid"], "{:.1f}")
best = cells.iloc[0]
put("bestCell", f"{best.pipeline}\\,$\\times$\\,{best.edge}")
put("bestFOne", best.macro_f1_mean)
put("bestFOneSd", best.macro_f1_sd)
lo, hi = json.loads(best.macro_f1_boot_ci.replace("'", '"')) if isinstance(best.macro_f1_boot_ci, str) else best.macro_f1_boot_ci
put("bestFOneCiLo", lo); put("bestFOneCiHi", hi)
put("bestBalAcc", best.balanced_acc_mean); put("bestAuc", best.roc_auc_mean); put("bestKappa", best.kappa_mean)
put("bestAcc", best.accuracy_mean); put("bestEce", best.ece_mean); put("bestBrier", best.brier_mean)
for c, g in enumerate(("AD", "FTD", "CN")):
    put(f"bestSens{g}", best[f"sens_{c}_mean"]); put(f"bestSpec{g}", best[f"spec_{c}_mean"])
pw = pd.read_csv(nb07 / "tables" / "grid_pairwise_representation.csv")
put("nPairsWilcoxonSig", int((pw["wilcoxon_p_holm"] < 0.05).sum()), "{:d}")
put("nPairsNbSig", int((pw["nb_p_holm"] < 0.05).sum()), "{:d}")
put("nPairs", len(pw), "{:d}")
p34 = pw[(pw.a == "P3") & (pw.b == "P4")].iloc[0]
put("pThreeFourD", p34.cohens_d, "{:.2f}"); put("pThreeFourNbP", p_fmt(p34.nb_p_holm))
put("pThreeFourWilP", p_fmt(p34.wilcoxon_p_holm)); put("pThreeFourDiff", p34.mean_diff)
for _, r in cells.iterrows():
    put(f"cell{r.pipeline}{r.edge}", r.macro_f1_mean)
rep = summ["marginal_representation"]; edge = summ["marginal_edge"]
for k, v in rep.items():
    put(f"rep{k}", v)
for k, v in edge.items():
    put(f"edge{k}", v)
for row in summ["anova"]:
    src = {"pipeline": "Rep", "edge": "Edge"}.get(row["Source"], "Inter")
    put(f"anova{src}F", row["F"], "{:.2f}")
    put(f"anova{src}DfOne", row["ddof1"], "{:d}"); put(f"anova{src}DfTwo", row["ddof2"], "{:d}")
    put(f"anova{src}P", p_fmt(row["p-GG-corr"]))
    put(f"anova{src}Eta", row["np2"], "{:.2f}")
put("friedmanP", p_fmt(summ["friedman_p"])); put("cdValue", summ["cd"], "{:.2f}")

# ---- leakage, permutation, ablations, baselines ----
leak = pd.read_csv(nb07 / "tables" / "leakage.csv")
for cell, g in leak.groupby("cell"):
    g = g.set_index("protocol")
    c = cell.replace("x", "")
    put(f"leak{c}CorrectSubj", g.loc["subject-level (correct)", "subject_macro_f1"])
    put(f"leak{c}LeakySubj", g.loc["window-level (leaky)", "subject_macro_f1"])
    put(f"leak{c}LeakyWin", g.loc["window-level (leaky)", "window_macro_f1"])
    put(f"leak{c}Inflation", g["inflation_window_vs_correct"].iloc[0])
for cell, r in summ["permutation"].items():
    c = cell.replace("x", "")
    put(f"perm{c}P", p_fmt(r["p"]).lstrip("=<")); put(f"perm{c}NullMean", r["null_mean"])
    put(f"perm{c}NullMax", r["null_max"]); put(f"perm{c}Obs", r["observed_seed0_macro_f1"])
    put(f"perm{c}N", r["n_perm"], "{:d}")
abl = pd.read_csv(nb07 / "tables" / "ablations.csv")
for _, r in abl.iterrows():
    c = r.cell.replace("x", "")
    k = f"abl{c}{r.ablation}"
    put(k + "Delta", r.delta, "{:+.3f}")
    put(k + "Lo", r.delta_ci_lo, "{:+.3f}"); put(k + "Hi", r.delta_ci_hi, "{:+.3f}")
    put(k + "P", p_fmt(r.nb_p_holm))
demo = pd.read_csv(REP / "nb03a_demographics.csv").groupby("model")["macro_f1"].agg(["mean", "std"])
put("demoFOne", demo.loc["demographics_lr", "mean"]); put("chanceFOne", demo.loc["chance_stratified", "mean"])

# early stopping: epoch of the best validation score in the main grid
runs = pd.DataFrame([json.loads(line) for f in sorted((REP / "grid").glob("runs_s*.jsonl")) for line in open(f)])
runs = runs[runs.tag == "main"]
put("bestEpochMedian", runs["best_epoch"].median(), "{:.0f}")
put("bestEpochLeThree", 100 * (runs["best_epoch"] <= 3).mean(), "{:.0f}")
for p, v in runs.groupby("pipeline")["best_epoch"].median().items():
    put(f"bestEpoch{p}", v, "{:.0f}")

# ---- must-fix re-runs: confirmation folds (seeds 5-9) and matched-training comparisons ----
conf = summ["confirmation"]
ct = nb07 / "tables" / "confirm"
ccells = pd.read_csv(ct / "grid_cells.csv").set_index(["pipeline", "edge"])
sel = ccells.loc[(best.pipeline, best.edge)]
put("confFOne", conf["selected_on_confirm_f1"]); put("confFOneSd", sel.macro_f1_sd)
put("confFOneCiLo", conf["selected_on_confirm_boot_ci"][0]); put("confFOneCiHi", conf["selected_on_confirm_boot_ci"][1])
put("confBalAcc", sel.balanced_acc_mean); put("confAuc", sel.roc_auc_mean); put("confKappa", sel.kappa_mean)
cb = ccells.index[0]
put("confBestCell", f"{cb[0]}\\,$\\times$\\,{cb[1]}"); put("confBestFOne", ccells.iloc[0].macro_f1_mean)
put("confRankRho", conf["rank_spearman_main_vs_confirm"], "{:.2f}")
cpw = pd.read_csv(ct / "grid_pairwise_representation.csv")
put("cPairsWilcoxonSig", int((cpw["wilcoxon_p_holm"] < 0.05).sum()), "{:d}")
put("cPairsNbSig", int((cpw["nb_p_holm"] < 0.05).sum()), "{:d}")
ccd = json.loads((ct / "grid_cd.json").read_text())
put("cFriedmanP", p_fmt(ccd["friedman_p"])); put("cCdValue", ccd["cd"], "{:.2f}")
for k, v in conf["confirm_marginal_representation"].items():
    put(f"crep{k}", v)
for k, v in conf["confirm_marginal_edge"].items():
    put(f"cedge{k}", v)
for row in conf["confirm_anova"]:
    src = {"pipeline": "Rep", "edge": "Edge"}.get(row["Source"], "Inter")
    put(f"canova{src}F", row["F"], "{:.2f}"); put(f"canova{src}P", p_fmt(row["p-GG-corr"]))
    put(f"canova{src}Eta", row["np2"], "{:.2f}")
for cell, r in conf["confirm_permutation"].items():
    c = cell.replace("x", "")
    put(f"cperm{c}P", p_fmt(r["p"]).lstrip("=<")); put(f"cperm{c}NullMean", r["null_mean"])
    put(f"cperm{c}NullMax", r["null_max"]); put(f"cperm{c}Obs", r["observed_seed0_macro_f1"])
    put(f"cperm{c}N", r["n_perm"], "{:d}")
cleak = pd.read_csv(ct / "leakage.csv")
for cell, g in cleak.groupby("cell"):
    g = g.set_index("protocol")
    c = cell.replace("x", "")
    put(f"cleak{c}CorrectSubj", g.loc["subject-level (correct)", "subject_macro_f1"])
    put(f"cleak{c}LeakySubj", g.loc["window-level (leaky)", "subject_macro_f1"])
    put(f"cleak{c}LeakyWin", g.loc["window-level (leaky)", "window_macro_f1"])
cabl = pd.read_csv(ct / "ablations.csv")
for _, r in cabl.iterrows():
    k = f"cabl{r.cell.replace('x', '')}{r.ablation}"
    put(k + "Delta", r.delta, "{:+.3f}")
    put(k + "Lo", r.delta_ci_lo, "{:+.3f}"); put(k + "Hi", r.delta_ci_hi, "{:+.3f}")
    put(k + "P", p_fmt(r.nb_p_holm)); put(k + "WilP", p_fmt(r.wilcoxon_p_holm))
put("cablNSigNb", int((cabl.nb_p_holm < 0.05).sum()), "{:d}")
put("cablNSigWil", int((cabl.wilcoxon_p_holm < 0.05).sum()), "{:d}")
fair = pd.read_csv(ct / "fair_comparisons.csv")
for _, r in fair.iterrows():
    k = f"fair{'Tr' if r.training_subjects == 'train only' else 'Tv'}{r.representation}"
    put(k + "Delta", r.delta, "{:+.3f}"); put(k + "P", p_fmt(r.nb_p_holm)); put(k + "WilP", p_fmt(r.wilcoxon_p_holm))
    put(k + "Gcn", r.gcn_f1); put(k + "Classical", r.classical_f1)
for regime, tag in (("train only", "Tr"), ("train + val", "Tv")):
    f = fair[fair.training_subjects == regime]
    put(f"fair{tag}NSig", int((f.nb_p_holm < 0.05).sum()), "{:d}")
    put(f"fair{tag}NSigWil", int((f.wilcoxon_p_holm < 0.05).sum()), "{:d}")
    put(f"fair{tag}DeltaMin", f.delta.min(), "{:+.3f}"); put(f"fair{tag}DeltaMax", f.delta.max(), "{:+.3f}")
refit = pd.DataFrame([json.loads(line) for f in sorted((REP / "confirm").glob("refit_runs_s*.jsonl")) for line in open(f)])
refit["macro_f1"] = refit["subject"].map(lambda s: s["macro_f1"])
rcell = refit.groupby(["pipeline", "edge"])["macro_f1"].mean()
put("refitBestFOne", rcell.loc[(best.pipeline, best.edge)])
put("refitMeanGain", (rcell - cells.set_index(["pipeline", "edge"])["macro_f1_mean"]).mean(), "{:+.3f}")
ctrain = pd.read_csv(REP / "confirm" / "classical_train.csv")
ctv = pd.read_csv(REP / "nb03c_classical.csv")
put("classicalTrBest", ctrain.groupby(["pipeline", "model"])["macro_f1"].mean().max())
put("classicalTvBest", ctv.groupby(["pipeline", "model"])["macro_f1"].mean().max())
crun = pd.concat([pd.read_json(f, lines=True) for f in sorted((REP / "confirm").glob("*.jsonl"))])
put("nConfirmRuns", len(crun), "{:,}"); put("cpuHoursConfirm", crun["seconds"].sum() / 3600, "{:.1f}")
put("nConfirmGridRuns", int((crun.tag == "confirm").sum()), "{:,}")
put("nRefitRuns", int((crun.tag == "refit").sum()), "{:,}")
put("nConfirmAblRuns", int((~crun.tag.isin(["confirm", "refit"])).sum()), "{:,}")

# ---- PSWE (RQ6) and the exploratory relative analysis ----
r6 = json.loads((REP / "rq6" / "rq6_pswe_stats.json").read_text())
for g in ("AD", "FTD", "CN"):
    put(f"psweRate{g}", r6["descriptives"]["rate_per_min"][g]["median"], "{:.2f}")
put("psweKwP", p_fmt(r6["rate_per_min"]["p"]))
for m, tag in (("nb_glm_adjusted", "Adj"), ("nb_glm_adjusted_slowing", "AdjSlow")):
    for grp, key in (("AD", AD), ("FTD", FTD)):
        put(f"pswe{tag}{grp}RR", r6[m]["rate_ratio"][key], "{:.2f}")
        put(f"pswe{tag}{grp}Lo", r6[m]["ci95"][key][0], "{:.2f}")
        put(f"pswe{tag}{grp}Hi", r6[m]["ci95"][key][1], "{:.2f}")
        put(f"pswe{tag}{grp}P", p_fmt(r6[m]["p"][key]))
put("psweSlowRho", r6["pswe_vs_slowing_spearman"]["rho"], "{:.2f}")
put("psweSlowPartial", r6["pswe_vs_slowing_partial_rank"]["r"], "{:.2f}")
put("psweMmseRho", r6["pswe_vs_mmse_patients"]["rho"], "{:.2f}")
put("psweMmseP", p_fmt(r6["pswe_vs_mmse_patients"]["p"]))
rel = json.loads((REP / "rq6_relative" / "rq6_relative_pswe_stats.json").read_text())
for v in ("drop2", "drop1.5", "drop3", "mad2", "mad3"):
    for m, tag in (("nb_glm_adjusted", "Adj"), ("nb_glm_adjusted_background_mpf", "AdjBg")):
        for grp, key in (("AD", AD), ("FTD", FTD)):
            k = f"rel{v.replace('.', 'p')}{tag}{grp}"
            put(k + "RR", rel[v][m]["rate_ratio"][key], "{:.2f}")
            put(k + "Lo", rel[v][m]["ci95"][key][0], "{:.2f}")
            put(k + "Hi", rel[v][m]["ci95"][key][1], "{:.2f}")
            put(k + "P", p_fmt(rel[v][m]["p"][key]))
    put(f"rel{v.replace('.', 'p')}SlowRho", rel[v]["pswe_vs_slowing_spearman"]["rho"], "{:.2f}")
    put(f"rel{v.replace('.', 'p')}MmseRho", rel[v]["pswe_vs_mmse_patients"]["rho"], "{:.2f}")

# ---- early-stopping sensitivity (NB10 -> NB07) ----
es_dir = nb07 / "tables" / "es_sensitivity"
if es_dir.exists():
    tag = {"patience 20 (main)": "Main", "patience 50": "PFifty", "fixed 100 epochs": "Fixed"}
    es = pd.read_csv(es_dir / "summary.csv")
    for _, r in es.iterrows():
        t = "es" + tag[r.rule]
        put(t + "BestCell", r.best_cell.replace("x", "\\,$\\times$\\,")); put(t + "BestFOne", r.best_f1)
        put(t + "PThreeSpatial", r.p3_spatial_f1); put(t + "Grand", r.grand_mean_f1)
        put(t + "RepF", r.rep_F, "{:.2f}"); put(t + "RepP", p_fmt(r.rep_p)); put(t + "RepEta", r.rep_np2, "{:.2f}")
        put(t + "EdgeF", r.edge_F, "{:.2f}"); put(t + "EdgeP", p_fmt(r.edge_p)); put(t + "InterP", p_fmt(r.inter_p))
        put(t + "Rho", r.spearman_vs_main, "{:.2f}"); put(t + "EpochMedian", r.median_best_epoch, "{:.0f}")
        put(t + "GvcSig", int(r.gcn_vs_classical_n_sig_wilcoxon), "{:d}")
    esr = pd.read_csv(es_dir / "representations.csv")
    for _, r in esr[esr.rule != "patience 20 (main)"].iterrows():
        t = "es" + tag[r.rule] + r.pipeline
        put(t + "Delta", r.delta_vs_main, "{:+.3f}"); put(t + "P", p_fmt(r.nb_p_holm)); put(t + "WilP", p_fmt(r.wilcoxon_p_holm))
    v = esr[esr.rule != "patience 20 (main)"]
    put("esNSigWil", int((v.wilcoxon_p_holm < 0.05).sum()), "{:d}"); put("esNSigNb", int((v.nb_p_holm < 0.05).sum()), "{:d}")
    rows = []
    for _, r in es.iterrows():
        bc = r.best_cell.replace("x", " $\\times$ ")
        rows.append(f"{r.rule} & {r.median_best_epoch:.0f} & {bc} & "
                    f"{r.p3_spatial_f1:.3f} & {r.grand_mean_f1:.3f} & {r.rep_F:.2f} ({p_txt_(r.rep_p)}) & "
                    f"{r.edge_F:.2f} ({p_txt_(r.edge_p)}) & {r.spearman_vs_main:.2f} \\\\")
    ES_ROWS = "\n".join(rows)
    rows = []
    for p_, g in esr.groupby("pipeline"):
        g = g.set_index("rule")
        cols = [f"{g.loc['patience 20 (main)', 'macro_f1']:.3f}"]
        for rule in ("patience 50", "fixed 100 epochs"):
            cols.append(f"{g.loc[rule, 'macro_f1']:.3f} & {g.loc[rule, 'delta_vs_main']:+.3f} & "
                        f"{g.loc[rule, 'wilcoxon_p_holm']:.2f}")
        rows.append(f"{p_} & {' & '.join(cols)} \\\\")
    ES_REP_ROWS = "\n".join(rows)
else:
    ES_ROWS = ES_REP_ROWS = None

# ---- external validation on ds004584 (NB11 -> NB07) ----
ext_dir = nb07 / "tables" / "external"
EXT_ROWS = None
if (ext_dir / "external_summary.json").exists():
    ex = json.loads((ext_dir / "external_summary.json").read_text())
    tr = ex["transfer"]
    for lvl, tag in (("cells", "Cells"), ("representations", "Reps")):
        put(f"ext{tag}Rho", tr[lvl]["spearman_rho"], "{:.2f}"); put(f"ext{tag}RhoP", p_fmt(tr[lvl]["spearman_p"]))
        put(f"ext{tag}Tau", tr[lvl]["kendall_tau"], "{:.2f}")
    s = ex["selected"]
    put("extSelFOne", s["macro_f1"]); put("extSelFOneSd", s["macro_f1_sd"]); put("extSelRank", s["rank"], "{:d}")
    put("extSelCiLo", s["boot_ci"][0]); put("extSelCiHi", s["boot_ci"][1])
    put("extSelBalAcc", s["balanced_acc"]); put("extSelAuc", s["roc_auc"]); put("extSelKappa", s["kappa"])
    put("extBestCell", ex["best_cell"]["cell"].replace("x", "\\,$\\times$\\,")); put("extBestFOne", ex["best_cell"]["macro_f1"])
    put("extDemoFOne", ex["demographics_f1"]); put("extChanceFOne", ex["chance_f1"])
    for row in ex["anova"]:
        src = {"pipeline": "Rep", "edge": "Edge"}.get(row["Source"], "Inter")
        put(f"extAnova{src}F", row["F"], "{:.2f}"); put(f"extAnova{src}P", p_fmt(row["p-GG-corr"]))
        put(f"extAnova{src}Eta", row["np2"], "{:.2f}")
    for k, v in ex["marginal_representation"].items():
        put(f"extRep{k}", v)
    for k, v in ex["marginal_edge"].items():
        put(f"extEdge{k}", v)
    for r in ex["leakage"]:
        if r["protocol"].startswith("window"):
            put("extLeakySubj", r["subject_macro_f1"]); put("extLeakyWin", r["window_macro_f1"])
    for cell, r in ex["permutation"].items():
        put("extPermP", p_fmt(r["p"]).lstrip("=<")); put("extPermN", r["n_perm"], "{:d}")
        put("extPermObs", r["observed_seed0_macro_f1"]); put("extPermNullMax", r["null_max"])
    g = pd.DataFrame(ex["gcn_vs_classical_train"])
    put("extGvcNSig", int((g.wilcoxon_p_holm < 0.05).sum()), "{:d}")
    put("extGvcDeltaMin", g.delta.min(), "{:+.3f}"); put("extGvcDeltaMax", g.delta.max(), "{:+.3f}")
    eq = json.loads((REP / "external" / "qc_report.json").read_text())
    put("extNWindows", eq["n_windows"], "{:,}"); put("extIcaExcluded", eq["ica_excluded"]["mean"], "{:.1f}")
    for gname in ("PD", "CN"):
        put(f"extThetaAlpha{gname}", eq["theta_alpha_ratio"][gname], "{:.2f}")
    e6 = json.loads((REP / "external" / "rq6" / "rq6_pswe_stats.json").read_text())
    PD = "C(group, Treatment('CN'))[T.PD]"
    for gname in ("PD", "CN"):
        put(f"extPsweRate{gname}", e6["descriptives"]["rate_per_min"][gname]["median"], "{:.2f}")
    for m, tag in (("nb_glm_adjusted", "Adj"), ("nb_glm_adjusted_slowing", "AdjSlow")):
        put(f"extPswe{tag}RR", e6[m]["rate_ratio"][PD], "{:.2f}"); put(f"extPswe{tag}P", p_fmt(e6[m]["p"][PD]))
        put(f"extPswe{tag}Lo", e6[m]["ci95"][PD][0], "{:.2f}"); put(f"extPswe{tag}Hi", e6[m]["ci95"][PD][1], "{:.2f}")
    put("extPsweSlowRho", e6["pswe_vs_slowing_spearman"]["rho"], "{:.2f}")
    put("extPsweMocaRho", e6["pswe_vs_mmse_patients"]["rho"], "{:.2f}"); put("extPsweMocaP", p_fmt(e6["pswe_vs_mmse_patients"]["p"]))
    er = json.loads((REP / "external" / "rq6_relative" / "rq6_relative_pswe_stats.json").read_text())
    put("extRelDropTwoRR", er["drop2"]["nb_glm_adjusted"]["rate_ratio"][PD], "{:.2f}")
    put("extRelDropTwoP", p_fmt(er["drop2"]["nb_glm_adjusted"]["p"][PD]))
    rel_p = [er[v]["nb_glm_adjusted"]["p"][PD] for v in er if isinstance(er[v], dict) and "nb_glm_adjusted" in er[v]]
    put("extRelMinP", p_fmt(min(rel_p)))
    ecells = pd.read_csv(ext_dir / "grid_cells.csv").set_index(["pipeline", "edge"])
    src_rep = cells.groupby("pipeline")["macro_f1_mean"].mean()
    ext_rep = ecells.groupby("pipeline")["macro_f1_mean"].mean()
    rows = []
    for p in sorted(REPN_ := {"P1": "P1 spectral", "P2": "P2 connectivity", "P3": "P3 wavelet", "P4": "P4 time domain",
                              "P5": "P5 raw", "P6": "P6 fused", "P7": "P7 PSWE"}):
        rows.append(f"{REPN_[p]} & {src_rep[p]:.3f} & {int(src_rep.rank(ascending=False)[p])} & {ext_rep[p]:.3f} & "
                    f"{int(ext_rep.rank(ascending=False)[p])} \\\\")
    EXT_ROWS = "\n".join(rows)

# ---- subject-level mixed model (NB13; primary inference) ----
MIX_ROWS = None
mix = REP / "mixed"
if (mix / "selection_summary.json").exists():
    sets = [(s, t) for s, t in (("selection", "Sel"), ("confirmation", "Conf"), ("external", "Ext"))
            if (mix / f"{s}_summary.json").exists()]
    summ_mix = {s: json.loads((mix / f"{s}_summary.json").read_text()) for s, _ in sets}
    for s, t in sets:
        m = summ_mix[s]
        put(f"mix{t}NRows", m["n_rows"], "{:,}"); put(f"mix{t}NScores", m["n_scores"], "{:,}")
        put(f"mix{t}NPairsSig", m["n_pairs_sig_holm"], "{:d}")
        for r in m["terms"]:
            k = {"representation": "Rep", "graph": "Edge", "interaction": "Inter", "true class": "Class"}[r["term"]]
            put(f"mix{t}{k}Chi", r["chi2"], "{:.1f}"); put(f"mix{t}{k}Df", r["df"], "{:d}"); put(f"mix{t}{k}P", p_fmt(r["p"]))
            put(f"mix{t}{k}F", r["F"], "{:.2f}"); put(f"mix{t}{k}DenDf", r["den_df"], "{:d}")
        for comp, v in m["variance_components"].items():
            put(f"mix{t}Share{comp.capitalize()}", 100 * v["share"], "{:.0f}")
        mg = pd.read_csv(mix / f"{s}_marginal.csv")
        for _, r in mg.iterrows():
            put(f"mix{t}M{r.level}", r["mean"])
    rows = []
    for term, label in (("representation", "Representation"), ("graph", "Graph construction"),
                        ("interaction", "Interaction")):
        cols = []
        for s, _ in sets:
            r = next(x for x in summ_mix[s]["terms"] if x["term"] == term)
            cols.append(f"{r['F']:.2f} ({r['df']}, {r['den_df']}) & {p_txt_(r['p'])}")
        rows.append(f"{label} & {' & '.join(cols)} \\\\")
    MIX_ROWS = "\n".join(rows)

# ---- A8 without ICA: fixed-threshold PSWE statistics ----
a8 = REP / "a7_a8" / "rq6_no_ica" / "rq6_pswe_stats.json"
if a8.exists():
    r8 = json.loads(a8.read_text())
    for grp, key in (("AD", AD), ("FTD", FTD)):
        put(f"noIcaPswe{grp}RR", r8["nb_glm_adjusted"]["rate_ratio"][key], "{:.2f}")
        put(f"noIcaPswe{grp}P", p_fmt(r8["nb_glm_adjusted"]["p"][key]))
        put(f"noIcaPsweSlow{grp}RR", r8["nb_glm_adjusted_slowing"]["rate_ratio"][key], "{:.2f}")
    put("noIcaPsweSlowRho", r8["pswe_vs_slowing_spearman"]["rho"], "{:.2f}")

qc = json.loads((REP / "nb02_qc_report.json").read_text())
put("nWindows", qc["n_windows"], "{:,}"); put("icaExcludedMean", qc["ica_excluded"]["mean"], "{:.1f}")
for g in ("AD", "FTD", "CN"):
    put(f"thetaAlpha{g}", qc["theta_alpha_ratio"][g], "{:.2f}")

# ---- paper tables (booktabs), from the same committed CSVs ----
TAB = OUT.parent / "tab"
TAB.mkdir(parents=True, exist_ok=True)
REPN = {"P1": "P1 spectral", "P2": "P2 connectivity", "P3": "P3 wavelet", "P4": "P4 time domain",
        "P5": "P5 raw", "P6": "P6 fused", "P7": "P7 PSWE"}


def ci(s):
    lo_, hi_ = json.loads(s) if isinstance(s, str) else s
    return f"[{lo_:.3f}, {hi_:.3f}]"


rows = []
for _, r in cells.sort_values(["pipeline", "edge"]).iterrows():
    c = ccells.loc[(r.pipeline, r.edge)]
    rows.append(f"{REPN[r.pipeline]} & {r.edge} & {r.macro_f1_mean:.3f} $\\pm$ {r.macro_f1_sd:.3f} & "
                f"{ci(r.macro_f1_boot_ci)} & {r.balanced_acc_mean:.3f} & {r.roc_auc_mean:.3f} & "
                f"{r.kappa_mean:.3f} & {r.brier_mean:.3f} & {c.macro_f1_mean:.3f} $\\pm$ {c.macro_f1_sd:.3f} \\\\")
(TAB / "cells.tex").write_text("\n".join(rows), encoding="utf-8")

# Table 4: every model under both training regimes (train only / train + val), same 25 test splits
mname = {"lr": "Logistic regression", "rf": "Random forest", "svm_rbf": "RBF-SVM"}


def ms(v):
    return f"{v.mean():.3f} $\\pm$ {v.std():.3f}"


rows = []
for p in sorted(REPN):
    g = runs[runs.pipeline == p]
    g = g.assign(macro_f1=g["subject"].map(lambda s: s["macro_f1"]))
    e_tr = g.groupby("edge")["macro_f1"].mean().idxmax()
    r = refit[refit.pipeline == p]
    e_tv = r.groupby("edge")["macro_f1"].mean().idxmax()
    rows.append(f"{REPN[p]} & GCN ({e_tr} / {e_tv}) & {ms(g[g.edge == e_tr].macro_f1)} & "
                f"{ms(r[r.edge == e_tv].macro_f1)} \\\\")
    for m in ("lr", "svm_rbf", "rf"):
        a = ctrain[(ctrain.pipeline == p) & (ctrain.model == m)]["macro_f1"]
        b = ctv[(ctv.pipeline == p) & (ctv.model == m)]["macro_f1"]
        if len(a) or len(b):
            rows.append(f" & {mname[m]} & {ms(a) if len(a) else '--'} & {ms(b) if len(b) else '--'} \\\\")
    rows.append("\\addlinespace")
d = pd.read_csv(REP / "nb03a_demographics.csv")
rows.append(f"Age + sex & Logistic regression & -- & {ms(d[d.model == 'demographics_lr'].macro_f1)} \\\\")
rows.append(f" & Stratified chance & -- & {ms(d[d.model == 'chance_stratified'].macro_f1)} \\\\")
(TAB / "baselines.tex").write_text("\n".join(rows), encoding="utf-8")

aname = {"A1_no_graph": "No graph (identity)", "A2_random_graph": "Random graph (degree-preserving)",
         "A3_k3": "$k=3$", "A3_k6": "$k=6$", "A3_full": "Fully connected", "A4_binary": "Binarised edges",
         "A5_depth1": "Depth 1", "A5_depth3": "Depth 3", "A6_gat": "GAT instead of GCN",
         "A9_unweighted": "No subject-equal weighting", "A12_residualized": "Age/sex regressed out",
         "RQ7_plus_P7": "+ P7 PSWE features", "RQ7_pswe_edge": "PSWE co-occurrence edges",
         "A7_window5": r"5\,s windows", "A7_window20": r"20\,s windows", "A8_no_ica": "No ICA artefact removal"}
def abl_rows(table):
    out = []
    for name_, g in table.groupby("ablation"):
        g = g.set_index("cell")
        cols = []
        for cell in ("P3xspatial", "P3xhybrid"):
            r = g.loc[cell]
            cols.append(f"{r.delta:+.3f} [{r.delta_ci_lo:+.3f}, {r.delta_ci_hi:+.3f}] & {r.nb_p_holm:.2f}")
        out.append(f"{aname[name_]} & {' & '.join(cols)} \\\\")
    return "\n".join(out)


(TAB / "ablations.tex").write_text(abl_rows(cabl), encoding="utf-8")          # confirmation folds (primary)
(TAB / "ablations_main.tex").write_text(abl_rows(abl), encoding="utf-8")      # selection folds (supplementary)


def p_txt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


aov = pd.read_csv(nb07 / "tables" / "grid_anova.csv")
caov = pd.read_csv(ct / "grid_anova.csv").set_index("Source")
src = {"pipeline": "Representation", "edge": "Graph construction", "pipeline * edge": "Interaction"}
rows = []
for _, r in aov.iterrows():
    c = caov.loc[r.Source]
    rows.append(f"{src[r.Source]} & {int(r.ddof1)}, {int(r.ddof2)} & {r.F:.2f} & {r['eps-GG']:.2f} & "
                f"{p_txt(r['p-GG-corr'])} & {r.np2:.2f} & {c.F:.2f} & {c['eps-GG']:.2f} & "
                f"{p_txt(c['p-GG-corr'])} & {c.np2:.2f} \\\\")
(TAB / "anova.tex").write_text("\n".join(rows), encoding="utf-8")

frows = []
for regime, g in fair.groupby("training_subjects", sort=False):
    for _, r in g.iterrows():
        frows.append(f"{regime} & {REPN[r.representation]} & GCN ({r.gcn}) & {mname.get(r.classical, r.classical)} & "
                     f"{r.gcn_f1:.3f} & {r.classical_f1:.3f} & {r.delta:+.3f} & {r.wilcoxon_p_holm:.2f} & "
                     f"{r.nb_p_holm:.2f} \\\\")
    frows.append("\\addlinespace")
(TAB / "fair.tex").write_text("\n".join(frows[:-1]), encoding="utf-8")
if MIX_ROWS is not None:
    (TAB / "mixed.tex").write_text(MIX_ROWS, encoding="utf-8")
if EXT_ROWS is not None:
    (TAB / "external.tex").write_text(EXT_ROWS, encoding="utf-8")
if ES_ROWS is not None:
    (TAB / "es.tex").write_text(ES_ROWS, encoding="utf-8")
    (TAB / "es_rep.tex").write_text(ES_REP_ROWS, encoding="utf-8")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("% generated by paper/make_numbers.py — do not edit\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
print(f"wrote {len(macros)} macros to {OUT}")
