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
    macros[name(key)] = fmt.format(value) if not isinstance(value, str) else value


def p_fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"={p:.3f}"


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
gvc = pd.read_csv(nb07 / "tables" / "gcn_vs_classical.csv")
for _, r in gvc.iterrows():
    put(f"gvc{r.representation}Delta", r.delta, "{:+.3f}")
    put(f"gvc{r.representation}P", p_fmt(r.nb_p_holm))
    put(f"gvc{r.representation}Classical", r.classical_f1)
demo = pd.read_csv(REP / "nb03a_demographics.csv").groupby("model")["macro_f1"].agg(["mean", "std"])
put("demoFOne", demo.loc["demographics_lr", "mean"]); put("chanceFOne", demo.loc["chance_stratified", "mean"])

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

qc = json.loads((REP / "nb02_qc_report.json").read_text())
put("nWindows", qc["n_windows"], "{:,}"); put("icaExcludedMean", qc["ica_excluded"]["mean"], "{:.1f}")
for g in ("AD", "FTD", "CN"):
    put(f"thetaAlpha{g}", qc["theta_alpha_ratio"][g], "{:.2f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("% generated by paper/make_numbers.py — do not edit\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")
print(f"wrote {len(macros)} macros to {OUT}")
