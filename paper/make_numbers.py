"""Write paper/generated/numbers.tex: LaTeX macros for every number quoted in the text, read from the
committed result files (reports/), so the prose never contains a hand-typed result.

Usage: python paper/make_numbers.py   (after copying NB07 outputs into reports/nb07/)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
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
put("cablN", len(cabl), "{:d}")
small = cabl[cabl.ablation.isin(["A3_k3", "A3_k6", "A4_binary", "A5_depth1", "A5_depth3", "A9_unweighted"])]
put("cablSmallMax", small.delta.abs().max(), "{:.3f}")
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
extra = [f for d in ("es_sensitivity", "external") for f in sorted((REP / d).glob("*.jsonl"))]
xrun = pd.concat([pd.read_json(f, lines=True)[["seconds"]] for f in extra])
put("nExtraRuns", len(xrun), "{:,}"); put("cpuHoursExtra", xrun["seconds"].sum() / 3600, "{:.1f}")
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
    put("esGrandShiftMax", (es.grand_mean_f1 - es.grand_mean_f1.iloc[0]).abs().max(), "{:.3f}")
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
    extra_sets = [(s, t) for s, t in (("selection_correct", "SelCorr"), ("confirmation_correct", "ConfCorr"))
                  if (mix / f"{s}_summary.json").exists()]
    summ_mix = {s: json.loads((mix / f"{s}_summary.json").read_text()) for s, _ in sets + extra_sets}
    for s, t in sets + extra_sets:
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
        e_ = mg[mg.factor == "edge"]["mean"]
        put(f"mix{t}EdgeRange", e_.max() - e_.min(), "{:.3f}")
        for r in m.get("rm_anova", []):                     # by-subject RM-ANOVA cross-check (GG-corrected)
            k = {"pipeline": "Rep", "edge": "Edge"}.get(r["Source"], "Inter")
            put(f"mix{t}Aov{k}F", r["F"], "{:.2f}"); put(f"mix{t}Aov{k}P", p_fmt(r["p-GG-corr"]))
    rows = []
    for term, label in (("representation", "Representation"), ("graph", "Graph construction"),
                        ("interaction", "Interaction")):
        cols = []
        for s, _ in sets:
            r = next(x for x in summ_mix[s]["terms"] if x["term"] == term)
            cols.append(f"{r['F']:.2f} ({r['df']}, {r['den_df']}) & {p_txt_(r['p'])}")
        rows.append(f"{label} & {' & '.join(cols)} \\\\")
    MIX_ROWS = "\n".join(rows)

# ---- deep-learning baselines (NB06 -> NB07) ----
DEEP = None
if (nb07 / "tables" / "deep" / "deep_baselines.csv").exists():
    DEEP = pd.read_csv(nb07 / "tables" / "deep" / "deep_baselines.csv")
    for model, tag in (("eegnet", "Eegnet"), ("cnn1d", "Cnn")):
        g = DEEP[DEEP.model == model]
        put(f"deep{tag}FOne", g.macro_f1.iloc[0]); put(f"deep{tag}Sd", g.sd.iloc[0])
        put(f"deep{tag}BalAcc", g.balanced_acc.iloc[0]); put(f"deep{tag}Auc", g.roc_auc.iloc[0])
        for _, r in g.iterrows():
            k = f"deep{tag}Vs{'Raw' if r.reference.startswith('GCN P5') else 'Best'}"
            put(k + "Delta", r.delta, "{:+.3f}"); put(k + "P", p_fmt(r.nb_p_holm)); put(k + "WilP", p_fmt(r.wilcoxon_p_holm))
            put(k + "RefFOne", r.reference_f1)
    put("deepNSigWil", int((DEEP.wilcoxon_p_holm < 0.05).sum()), "{:d}")
    put("deepNSigNb", int((DEEP.nb_p_holm < 0.05).sum()), "{:d}")
    sec = sum(json.loads(line)["seconds"] for f in sorted((REP / "deep").glob("deep_runs_s*.jsonl")) for line in open(f))
    put("deepGpuHours", sec / 3600, "{:.2f}")

# ---- PSWE robustness: zero-padded MPF + artefact rejection (NB14) ----
for ds, tag, groups in (("ds004504", "Rob", (("AD", AD), ("FTD", FTD))),
                        ("ds004584", "ExtRob", (("PD", "C(group, Treatment('CN'))[T.PD]"),))):
    f = REP / "pswe_robust" / f"robust_{ds}" / "rq6_robust_pswe_stats.json"
    if not f.exists():
        continue
    rb = json.loads(f.read_text())
    pr = rb[rb["primary"]]
    put(f"pswe{tag}Identical", "yes" if rb["original"]["identical_counts_to_main"] else "no")
    put(f"pswe{tag}RhoMain", pr["rate_vs_main_spearman"], "{:.2f}")
    put(f"pswe{tag}SlowRho", pr["pswe_vs_slowing_spearman"]["rho"], "{:.2f}")
    put(f"pswe{tag}RejectPct", 100 * max(pr["rejected_fraction_by_group"].values()), "{:.1f}")
    for grp, key in groups:
        for m, mt in (("nb_glm_adjusted", "Adj"), ("nb_glm_adjusted_slowing", "AdjSlow")):
            put(f"pswe{tag}{mt}{grp}RR", pr[m]["rate_ratio"][key], "{:.2f}")
            put(f"pswe{tag}{mt}{grp}P", p_fmt(pr[m]["p"][key]))

# ---- equivalence (TOST, +/-0.05, NB-corrected) and minimum detectable effects (NB15) ----
EQ_ROWS = GRID_ROWS = None
eqf = REP / "equivalence" / "equivalence.csv"
if eqf.exists():
    eq = pd.read_csv(eqf)
    fam = {"GCN vs non-graph twin": "Gvc", "graph construction": "Graph", "ablation": "Abl",
           "deep baseline": "Deep", "stopping rule": "Es", "learned adjacency": "Learn"}
    for f_, tag in fam.items():
        g = eq[eq.family == f_]
        put(f"eq{tag}N", len(g), "{:d}"); put(f"eq{tag}NEquiv", int(g.equivalent.sum()), "{:d}")
        put(f"eq{tag}Mde", g.mde80.median(), "{:.2f}"); put(f"eq{tag}MdeMax", g.mde80.max(), "{:.2f}")
        put(f"eq{tag}CiMax", max(g.ci90_lo.abs().max(), g.ci90_hi.abs().max()), "{:.3f}")
    gr = eq[eq.family == "graph construction"]
    put("eqGraphPMax", p_fmt(gr.p_tost_holm.max()))
    put("eqMargin", eq.margin.iloc[0], "{:.2f}")
    lab = {"GCN vs non-graph twin": "GCN vs non-graph twin", "graph construction": "Graph types (pairwise)",
           "ablation": "Ablations", "deep baseline": "Deep-learning baselines", "stopping rule": "Stopping rules",
           "learned adjacency": "Learned vs fixed graph"}
    rows = []
    for (f_, s_), g in eq.groupby(["family", "setting"], sort=False):
        rows.append(f"{lab[f_]} & {s_} & {len(g)} & {int(g.equivalent.sum())} & "
                    f"{g.delta.abs().max():.3f} & {g.mde80.median():.2f} \\\\")
    EQ_ROWS = "\n".join(rows)

# ---- learned adjacency (NB16): trainable graph initialised at the spatial graph ----
lfiles = sorted((REP / "learned").glob("learned_runs_s*.jsonl"))
if lfiles:
    L = pd.DataFrame([json.loads(line) for f in lfiles for line in open(f)])
    L["macro_f1"] = L["subject"].map(lambda s: s["macro_f1"])
    lm = L[L.tag == "learned_adj"].groupby("pipeline")["macro_f1"].mean()
    ref = runs.assign(macro_f1=runs["subject"].map(lambda s: s["macro_f1"]))
    rs = ref[ref.edge == "spatial"].groupby("pipeline")["macro_f1"].mean().reindex(lm.index)
    put("learnBestRep", lm.idxmax()); put("learnBestFOne", lm.max())
    put("learnDeltaMin", (lm - rs).min(), "{:+.3f}"); put("learnDeltaMax", (lm - rs).max(), "{:+.3f}")
    put("learnRho", lm.corr(rs, method="spearman"), "{:.2f}")
    put("learnPThree", lm.get("P3", np.nan))
    lc = L[L.tag == "learned_adj_confirm"]
    if len(lc):
        put("learnConfPThree", lc["macro_f1"].mean())
        cf = pd.DataFrame([json.loads(line) for f in sorted((REP / "confirm").glob("confirm_runs_s*.jsonl"))
                           for line in open(f)])
        cf = cf[(cf.tag == "confirm") & (cf.pipeline == "P3") & (cf.edge == "spatial")]
        if len(cf):
            put("learnConfDelta", lc["macro_f1"].mean() - cf["subject"].map(lambda s: s["macro_f1"]).mean(), "{:+.3f}")
    put("learnPThreeDelta", round(lm.get("P3", np.nan) - rs.get("P3", np.nan), 3) + 0.0, "{:+.3f}")
    put("learnNRuns", len(L), "{:d}")
    put("cpuHoursLearned", L["seconds"].sum() / 3600, "{:.1f}")

# ---- occlusion importance (NB16) ----
occ_f = REP / "explain" / "occlusion_summary.csv"
if occ_f.exists():
    occ = pd.read_csv(occ_f)
    occ["unit"] = occ["unit"].str.replace("-", "--", regex=False)
    el = occ[occ.kind == "electrode"].sort_values("drop", ascending=False)
    put("occTopElectrodes", ", ".join(el.unit.head(4)))
    put("occTopDrop", el["drop"].iloc[0], "{:.3f}"); put("occElecMedian", el["drop"].median(), "{:.3f}")
    put("occNElecPositive", int((el.ci_lo > 0).sum()), "{:d}")
    bd = occ[occ.kind == "band"].sort_values("drop", ascending=False)
    put("occTopBand", bd.unit.iloc[0]); put("occTopBandDrop", bd["drop"].iloc[0], "{:.3f}")
    put("occSecondBand", bd.unit.iloc[1]); put("occSecondBandDrop", bd["drop"].iloc[1], "{:.3f}")
    put("occBottomBand", bd.unit.iloc[-1]); put("occBottomBandDrop", bd["drop"].iloc[-1], "{:.3f}")
    put("occTopBandLo", bd.ci_lo.iloc[0], "{:.3f}"); put("occTopBandHi", bd.ci_hi.iloc[0], "{:.3f}")
    put("occNBandPositive", int((bd.ci_lo > 0).sum()), "{:d}")
    for cname in ("AD", "FTD", "CN"):
        put(f"occTop{cname}", el.sort_values(f"drop_{cname}", ascending=False).unit.iloc[0])
        b_ = bd.sort_values(f"drop_{cname}", ascending=False)
        put(f"occBand{cname}", b_.unit.iloc[0]); put(f"occBand{cname}Drop", b_[f"drop_{cname}"].iloc[0], "{:.3f}")
    occ_raw = REP / "explain" / "occlusion_raw.csv"
    if occ_raw.exists():
        oraw = pd.read_csv(occ_raw)
        put("occNSubjects", oraw["subject"].nunique(), "{:d}")

# ---- PSWE A11 rule grid and collinearity of the slowing adjustment (NB15) ----
for ds, tag, pats in (("ds004504", "", ("AD", "FTD")), ("ds004584", "Ext", ("PD",))):
    gd = REP / "pswe_grid" / f"pswe_grid_{ds}"
    if not (gd / "pswe_grid.csv").exists():
        continue
    gg = pd.read_csv(gd / "pswe_grid.csv")
    for p_ in pats:
        put(f"grid{tag}{p_}RrMin", gg[f"rr_adj_{p_}"].min(), "{:.2f}"); put(f"grid{tag}{p_}RrMax", gg[f"rr_adj_{p_}"].max(), "{:.2f}")
        put(f"grid{tag}{p_}NSig", int((gg[f"p_adj_{p_}"] < 0.05).sum()), "{:d}")
        put(f"grid{tag}{p_}SlowMax", gg[f"rr_adj_slow_{p_}"].max(), "{:.2f}")
        put(f"grid{tag}{p_}SlowMin", gg[f"rr_adj_slow_{p_}"].min(), "{:.2f}")
    put(f"grid{tag}N", len(gg), "{:d}")
    put(f"grid{tag}RhoMin", gg.slowing_rho.min(), "{:.2f}"); put(f"grid{tag}RhoMax", gg.slowing_rho.max(), "{:.2f}")
    co = json.loads((gd / "pswe_collinearity.json").read_text())
    put(f"col{tag}Vif", co["vif_slowing"], "{:.2f}")
    for p_ in pats:
        key = f"C(group, Treatment('CN'))[T.{p_}]"
        put(f"col{tag}{p_}Outside", 100 * co["slowing_outside_control_range"][p_], "{:.0f}")
        put(f"col{tag}{p_}InSupport", co["common_support"]["n_in_support"][p_], "{:d}")
        for m_, mt in (("adjusted", "Sup"), ("adjusted_slowing", "SupSlow")):
            r_ = co["common_support"][m_][key]
            put(f"col{tag}{mt}{p_}RR", r_["rate_ratio"], "{:.2f}"); put(f"col{tag}{mt}{p_}P", p_fmt(r_["p"]))
        put(f"col{tag}Oe{p_}", co["observed_expected"][p_], "{:.2f}")
    put(f"col{tag}OeCN", co["observed_expected"]["CN"], "{:.2f}")
    put(f"col{tag}OeP", p_fmt(co["observed_expected_kruskal_p"]))
    if ds == "ds004504":
        GRID_ROWS = "\n".join(
            f"{r.threshold_hz:.0f} & {r.min_duration_s:.0f} & {r.rr_adj_AD:.2f} & {r.rr_adj_FTD:.2f} & "
            f"{r.rr_adj_slow_AD:.2f} & {r.rr_adj_slow_FTD:.2f} & {r.slowing_rho:.2f} \\\\" for r in gg.itertuples())

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
    if p == "P5" and DEEP is not None:                       # deep-learning baselines on the same raw windows
        for model, label in (("eegnet", "EEGNet"), ("cnn1d", "1D-CNN")):
            r_ = DEEP[DEEP.model == model].iloc[0]
            rows.append(f" & {label} & {r_.macro_f1:.3f} $\\pm$ {r_.sd:.3f} & -- \\\\")
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
if EQ_ROWS is not None:
    (TAB / "equivalence.tex").write_text(EQ_ROWS, encoding="utf-8")
if GRID_ROWS is not None:
    (TAB / "pswe_grid.tex").write_text(GRID_ROWS, encoding="utf-8")
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
