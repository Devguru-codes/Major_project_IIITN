"""NB07: build every table and figure from the committed raw results (reports/) plus the
feature cache and preprocessed recordings. No number is typed by hand.
"""
from __future__ import annotations

import glob
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as F
from .stats_ablation import write_all
from .stats_grid import analyse_grid, load_runs, write_analysis


def build_report(reports: str | Path, cache_dir: str | Path, preproc_dirs: list[str], cfg: dict,
                 out_dir: str | Path, n_boot: int = 2000) -> dict:
    from .cache import FeatureCache
    from .preprocess import iter_preprocessed

    rep, out = Path(reports), Path(out_dir)
    figs, tabs = out / "figures", out / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    names = cfg["dataset"]["class_names"]
    errors = []

    def step(label, fn):
        try:
            fn()
            print(f"[report] ok   {label}", flush=True)
        except Exception:
            errors.append(label)
            print(f"[report] FAIL {label}\n{traceback.format_exc()}", flush=True)

    main = load_runs(sorted(glob.glob(str(rep / "grid" / "runs_s*.jsonl"))), tag="main")
    abl = load_runs(sorted(glob.glob(str(rep / "ablations" / "ablations_s*.jsonl"))))
    classical = pd.read_csv(rep / "nb03c_classical.csv")
    demo = pd.read_csv(rep / "nb03a_demographics.csv")
    subj = pd.read_csv(rep / "rq6" / "pswe_subject_table.csv")
    topo = pd.read_csv(rep / "rq6" / "pswe_topography.csv", index_col=0)

    grid = analyse_grid(main, n_boot=n_boot)
    write_analysis(grid, tabs)
    abl_res = write_all(main, abl, classical, demo, tabs)

    # LaTeX versions of the journal tables
    def latex():
        c = grid["cells"][["pipeline", "edge", "macro_f1_mean", "macro_f1_sd", "balanced_acc_mean", "roc_auc_mean",
                           "kappa_mean", "brier_mean"]].round(3)
        c.to_latex(tabs / "table3_cells.tex", index=False)
        abl_res["baselines"].round(3).to_latex(tabs / "table4_baselines.tex", index=False)
        abl_res["ablations"][["cell", "ablation", "delta", "delta_ci_lo", "delta_ci_hi", "nb_p_holm"]].round(3) \
            .to_latex(tabs / "table5_ablations.tex", index=False)
        grid["anova"].round(4).to_latex(tabs / "table6_anova.tex", index=False)
    step("latex tables", latex)

    cache = FeatureCache(cache_dir)
    step("fig02 cohort", lambda: F.fig_cohort(subj, figs))
    step("fig03 band topomaps", lambda: F.fig_band_topomaps(cache, cfg, figs))

    def pswe_example():
        have = {Path(f).stem: d for d in preproc_dirs for f in glob.glob(f"{d}/sub-*.npz")}
        ad = subj[(subj.label == 0) & subj.subject.isin(have)].sort_values("rate_per_min", ascending=False)
        s = ad.iloc[0]["subject"]
        _, data, sfreq = next(iter_preprocessed(have[s], [s]))
        ev_ch = pd.read_csv(Path(cache_dir) / "pswe_events.csv").query("subject == @s")["channel"].value_counts()
        F.fig_pswe_example(data, sfreq, cfg, cfg["dataset"]["channels"], ev_ch.index[0], figs)
    step("fig04 pswe example", pswe_example)
    step("fig05 pswe burden", lambda: F.fig_pswe_burden(subj, topo, cfg, figs))
    step("fig06/07 main effects", lambda: F.fig_main_effects(main, figs))
    step("fig08 interaction", lambda: F.fig_interaction(grid["cells"], figs))
    step("fig09 critical difference", lambda: F.fig_cd(grid["cd"], figs))
    step("fig10 confusion matrices", lambda: F.fig_confusions(main, names, figs))
    top = [tuple(r) for r in grid["cells"][["pipeline", "edge"]].head(3).to_numpy()]
    step("fig11 roc/pr", lambda: F.fig_roc_pr(main, top, names, figs))
    step("fig12 leakage", lambda: F.fig_leakage(abl_res["leakage"], figs))
    step("fig13 ablation forest", lambda: F.fig_forest(abl_res["ablations"], figs))
    step("figS3 seed variance", lambda: F.fig_seed_variance(main, figs))
    step("figS5 permutation", lambda: F.fig_permutation(abl_res["permutation"], figs))

    # ---- must-fix re-runs (present once reports/confirm/ is committed) ----
    conf_dir = rep / "confirm"
    if conf_dir.exists():
        from .stats_ablation import ablation_table, fair_comparisons, leakage_table, permutation_test

        cfiles = sorted(glob.glob(str(conf_dir / "confirm_runs_s*.jsonl")))
        confirm = load_runs(cfiles, tag="confirm").assign(tag="main")      # analyse like a main grid
        cgrid = analyse_grid(confirm, n_boot=n_boot)
        write_analysis(cgrid, tabs / "confirm")
        cabl = load_runs(sorted(glob.glob(str(conf_dir / "ablations_confirm_s*.jsonl"))) +
                         sorted(glob.glob(str(conf_dir / "perm_confirm_s*.jsonl"))))
        conf_res = {"ablations": ablation_table(confirm, cabl), "leakage": leakage_table(confirm, cabl)}
        refit = load_runs(sorted(glob.glob(str(conf_dir / "refit_runs_s*.jsonl"))), tag="refit")
        ctrain = pd.read_csv(conf_dir / "classical_train.csv")
        conf_res["fair_comparisons"] = fair_comparisons(main, refit, ctrain, classical)
        for k, v in conf_res.items():
            v.to_csv(tabs / "confirm" / f"{k}.csv", index=False)
        conf_res["permutation"] = permutation_test(confirm, cabl)
        (tabs / "confirm" / "permutation.json").write_text(json.dumps(conf_res["permutation"], indent=1))
        # selection-free estimate: the cell chosen on seeds 0-4, evaluated on seeds 5-9
        chosen = grid["cells"].iloc[0]
        sel = cgrid["cells"].set_index(["pipeline", "edge"]).loc[(chosen.pipeline, chosen.edge)]
        step("fig13 ablation forest (confirmation)",
             lambda: F.fig_forest(conf_res["ablations"], figs, name="fig13_ablation_forest"))
        step("fig12 leakage (confirmation)", lambda: F.fig_leakage(conf_res["leakage"], figs))
        step("figS5 permutation (confirmation)", lambda: F.fig_permutation(conf_res["permutation"], figs))
        step("figS6 main vs confirmation ranking",
             lambda: F.fig_confirmation(grid["cells"], cgrid["cells"], figs))
        confirmation_summary = {
            "selected_cell": f"{chosen.pipeline}x{chosen.edge}",
            "selected_on_main_f1": float(chosen.macro_f1_mean),
            "selected_on_confirm_f1": float(sel.macro_f1_mean),
            "selected_on_confirm_boot_ci": sel.macro_f1_boot_ci,
            "confirm_best_cell": f"{cgrid['cells'].iloc[0].pipeline}x{cgrid['cells'].iloc[0].edge}",
            "confirm_anova": cgrid["anova"].to_dict("records"),
            "confirm_marginal_representation": cgrid["marginal_representation"]["mean"].round(4).to_dict(),
            "confirm_marginal_edge": cgrid["marginal_edge"]["mean"].round(4).to_dict(),
            "rank_spearman_main_vs_confirm": float(
                grid["cells"].set_index(["pipeline", "edge"])["macro_f1_mean"].corr(
                    cgrid["cells"].set_index(["pipeline", "edge"])["macro_f1_mean"], method="spearman")),
            "confirm_permutation": {k: {kk: vv for kk, vv in v.items() if kk != "null"}
                                    for k, v in conf_res["permutation"].items()},
        }
    else:
        confirmation_summary = None

    # ---- early-stopping sensitivity (present once reports/es_sensitivity/ is committed) ----
    es_dir, es_summary = rep / "es_sensitivity", None
    if es_dir.exists():
        from .stats_sensitivity import ES_RULES, es_sensitivity, write_sensitivity

        es_runs = load_runs(sorted(glob.glob(str(es_dir / "es_runs_s*.jsonl"))))
        variants = {label: es_runs[es_runs.tag == tag] for tag, label in ES_RULES.items()}
        ctrain = pd.read_csv(conf_dir / "classical_train.csv")
        es = es_sensitivity(main, variants, ctrain)
        write_sensitivity(es, tabs / "es_sensitivity")
        step("figS7 early-stopping sensitivity", lambda: F.fig_es_sensitivity(
            es["representations"], {"patience 20 (main)": main, **variants}, figs))
        es_summary = es["summary"].to_dict("records")

    best = grid["cells"].iloc[0]
    summary = {
        "n_main_runs": int(len(main)), "n_ablation_runs": int(len(abl)),
        "best_cell": f"{best.pipeline}x{best.edge}", "best_macro_f1": float(best.macro_f1_mean),
        "best_macro_f1_boot_ci": best.macro_f1_boot_ci, "anova": grid["anova"].to_dict("records"),
        "friedman_p": grid["cd"]["friedman_p"], "cd": grid["cd"]["cd"],
        "marginal_representation": grid["marginal_representation"]["mean"].round(4).to_dict(),
        "marginal_edge": grid["marginal_edge"]["mean"].round(4).to_dict(),
        "leakage": abl_res["leakage"].round(4).to_dict("records"),
        "permutation": {k: {kk: vv for kk, vv in v.items() if kk != "null"} for k, v in abl_res["permutation"].items()},
        "cpu_hours_main_grid": float(main["seconds"].sum() / 3600),
        "confirmation": confirmation_summary,
        "es_sensitivity": es_summary,
        "report_errors": errors,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if errors:
        raise RuntimeError(f"report steps failed: {errors}")
    return summary
