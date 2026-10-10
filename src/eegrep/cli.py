"""Command line: `python -m eegrep <command>` (also installed as `eegrep`)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from .config import load_config


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def _shard_subjects(participants: str, cfg: dict, shard: int | None, n_shards: int | None) -> list[str]:
    from .data.participants import load_participants

    subjects = list(load_participants(participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])["subject"])
    return subjects if shard is None else subjects[shard::n_shards]


def cmd_download(a, cfg):
    from .data.download import download_dataset

    subjects = None
    if a.shard is not None:
        download_dataset(cfg["dataset"]["name"], a.dest, cfg["dataset"]["s3_bucket"], subjects=[])
        subjects = _shard_subjects(Path(a.dest) / "participants.tsv", cfg, a.shard, a.n_shards)
    download_dataset(cfg["dataset"]["name"], a.dest, cfg["dataset"]["s3_bucket"], subjects=subjects)


def cmd_preprocess(a, cfg):
    from .preprocess import preprocess_dataset

    subjects = _shard_subjects(Path(a.bids) / "participants.tsv", cfg, a.shard, a.n_shards)
    preprocess_dataset(a.bids, subjects, cfg, a.out, n_jobs=a.n_jobs)


def cmd_cache(a, cfg):
    from . import graphs
    from .cache import build_cache
    from .data.participants import load_participants
    from .preprocess import iter_preprocessed

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    subjects = sorted(p.stem for p in Path(a.preproc).glob("sub-*.npz"))
    labels = dict(zip(df["subject"], df["label"]))
    coords = graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])
    build_cache(iter_preprocessed(a.preproc, subjects), labels, cfg, coords, a.out)


def cmd_merge(a, cfg):
    from .cache import merge_caches

    merge_caches(a.inputs, a.out)


def cmd_qc(a, cfg):
    from .qc import gate_failures, qc_report

    rep = qc_report(a.cache, a.logs or [], cfg["dataset"]["class_names"])
    rep["gate_failures"] = gate_failures(rep, cfg["dataset"].get("qc"))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["gate_failures"] and not a.no_fail:
        raise SystemExit(f"QC gate failed: {rep['gate_failures']}")


def cmd_pswe_stats(a, cfg):
    from .data.participants import load_participants
    from .stats_pswe import write_rq6

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    res = write_rq6(a.cache, df, cfg["dataset"]["class_names"], a.out)
    print(json.dumps(res, indent=1))


def cmd_pswe_relative(a, cfg):
    from .data.participants import load_participants
    from .stats_pswe import write_relative

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    res = write_relative(a.preproc, a.cache, df, cfg, a.out)
    prim = res[res["primary"]]
    print(json.dumps({"primary": res["primary"], **{k: prim[k] for k in prim if k.startswith(("nb_", "pswe_v"))}},
                     indent=1))


def cmd_classical(a, cfg):
    import pandas as pd

    from .baselines import classical_baseline, summarize
    from .cache import FeatureCache
    from .splits import load_folds

    cache = FeatureCache(a.cache)
    folds = load_folds(a.folds, sorted(set(cache.subject)))
    res = pd.concat([classical_baseline(cache, folds, p, cfg, fit_on=a.fit_on) for p in a.pipelines.split(",")])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(summarize(res, by=("pipeline", "model")).round(3).to_string())


def cmd_ablate(a, cfg):
    from .ablations import ABLATIONS, run_ablation, run_permutations
    from .cache import FeatureCache
    from .data.participants import load_participants
    from .runner import ResultStore
    from .splits import load_folds

    cache = FeatureCache(a.cache)
    folds = load_folds(a.folds, sorted(set(cache.subject)))
    part = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    store = ResultStore(a.out)
    deadline = time.time() + 3600 * a.max_hours
    for cell in a.cells.split(","):
        pipeline, edge = cell.split("x")
        names = {"all": list(ABLATIONS), "none": []}.get(a.which, a.which.split(","))
        for name in names:
            run_ablation(name, cache, folds, cfg, store, pipeline=pipeline, edge=edge, participants=part,
                         deadline=deadline, device=a.device)
        if a.n_perm:
            run_permutations(cache, folds, cfg, store, pipeline=pipeline, edge=edge, n_perm=a.n_perm,
                             perm_start=a.perm_start, deadline=deadline, device=a.device)
    print(json.dumps({"runs_done": len(store.done)}))


def cmd_report(a, cfg):
    from .report import build_report

    res = build_report(a.reports, a.cache, a.preproc or [], cfg, a.out, n_boot=a.n_boot)
    print(json.dumps(res, indent=1, default=float))


def cmd_select_cells(a, cfg):
    from .ablations import select_cells

    print(",".join(select_cells(a.runs, a.top)))


def cmd_demographics(a, cfg):
    from .baselines import demographics_baseline, summarize
    from .data.participants import load_participants
    from .splits import load_folds

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    res = demographics_baseline(df, load_folds(a.folds, df["subject"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(summarize(res).round(3).to_string())


def cmd_folds(a, cfg):
    from .data.participants import load_participants
    from .splits import file_sha256, make_folds, save_folds

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    cv = cfg["cv"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cv["seeds"]
    folds = make_folds(df["subject"], df["label"], seeds, cv["n_folds"], cv["val_frac"],
                       meta={"dataset": cfg["dataset"]["cite_doi"],
                             "participants_sha256": file_sha256(a.participants)})
    save_folds(folds, a.out)
    print(f"[folds] {len(df)} subjects, {len(seeds)} repeats (seeds {seeds}) x {cv['n_folds']} folds -> {a.out}")


def cmd_smoke(a, cfg):
    """End-to-end on synthetic EEG: cache -> folds -> every pipeline and edge type -> resume check."""
    from . import graphs, synthetic
    from .cache import FeatureCache, build_cache
    from .data.participants import load_participants
    from .runner import ResultStore, run_cell, summarize
    from .splits import make_folds

    t0 = time.time()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    recs, part = synthetic.make_recordings()
    part.to_csv(work / "participants.tsv", sep="\t", index=False)
    df = load_participants(work / "participants.tsv", cfg["dataset"]["group_to_label"],
                           cfg["dataset"]["participant_columns"])
    labels = dict(zip(df["subject"], df["label"]))
    coords = graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])
    cache = FeatureCache(build_cache(recs, labels, cfg, coords, work / "cache"))
    folds = make_folds(df["subject"], df["label"], [0], cfg["cv"]["n_folds"], cfg["cv"]["val_frac"])
    store = ResultStore(work / "runs.jsonl")
    quick = {"train.max_epochs": 3}
    cells = [(p, "hybrid") for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]]
    cells += [("P1", e) for e in ["spatial", "functional", "pswe", "identity", "random"]]
    for p, e in cells:
        run_cell(cache, folds, cfg, store, pipeline=p, edge=e, tag="smoke", overrides=quick)
    rerun = run_cell(cache, folds, cfg, store, pipeline="P1", edge="hybrid", tag="smoke", overrides=quick)
    assert rerun == 0, "resume failed: finished runs were executed again"
    for k, (m, s, n) in sorted(summarize(work / "runs.jsonl").items()):
        print(f"[smoke] {k}: macro-F1 {m:.3f} ± {s:.3f} (n={n})")
    print(f"[smoke] OK in {time.time() - t0:.0f} s")


def cmd_bench(a, cfg):
    from .cache import FeatureCache
    from .runner import bench
    from .splits import load_folds

    cache = FeatureCache(a.cache)
    folds = load_folds(a.folds, sorted(set(cache.subject)))
    bench(cache, folds, cfg, pipeline=a.pipeline, edge=a.edge, n_runs_planned=a.n_runs,
          max_hours=a.max_hours, device=a.device, out=a.out)


def cmd_grid(a, cfg):
    from .cache import FeatureCache
    from .runner import ResultStore, run_cell
    from .splits import load_folds

    cache = FeatureCache(a.cache)
    folds = load_folds(a.folds, sorted(set(cache.subject)))
    store = ResultStore(a.out)
    deadline = time.time() + 3600 * (a.max_hours or cfg["budget"]["max_hours"])
    refit = None
    if a.refit_from:
        from .runner import best_epochs

        refit = best_epochs(a.refit_from)
    for p in a.pipelines.split(","):
        for e in a.edges.split(","):
            run_cell(cache, folds, cfg, store, pipeline=p, edge=e, tag=a.tag, deadline=deadline,
                     device=a.device, overrides=_overrides(a.set), refit_epochs=refit)
    print(json.dumps({"runs_done": len(store.done)}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eegrep")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download"); s.add_argument("--dest", required=True); s.set_defaults(fn=cmd_download)
    s.add_argument("--shard", type=int); s.add_argument("--n-shards", type=int)
    s = sub.add_parser("preprocess"); s.add_argument("--bids", required=True); s.add_argument("--out", required=True)
    s.add_argument("--shard", type=int); s.add_argument("--n-shards", type=int)
    s.add_argument("--n-jobs", type=int, default=4); s.set_defaults(fn=cmd_preprocess)
    s = sub.add_parser("cache"); s.add_argument("--preproc", required=True); s.add_argument("--participants", required=True)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_cache)
    s = sub.add_parser("merge-caches"); s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_merge)
    s = sub.add_parser("qc"); s.add_argument("--cache", required=True); s.add_argument("--logs", nargs="*")
    s.add_argument("--out", default="qc_report.json"); s.add_argument("--no-fail", action="store_true")
    s.set_defaults(fn=cmd_qc)
    s = sub.add_parser("pswe-stats"); s.add_argument("--cache", required=True)
    s.add_argument("--participants", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_pswe_stats)
    s = sub.add_parser("pswe-relative"); s.add_argument("--preproc", nargs="+", required=True)
    s.add_argument("--cache", required=True); s.add_argument("--participants", required=True)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_pswe_relative)
    s = sub.add_parser("classical");s.add_argument("--cache", required=True); s.add_argument("--folds", required=True)
    s.add_argument("--pipelines", default="P1,P2,P3,P4,P6,P7"); s.add_argument("--out", required=True)
    s.add_argument("--fit-on", choices=["train", "trainval"], default="trainval")
    s.set_defaults(fn=cmd_classical)
    s = sub.add_parser("ablate"); s.add_argument("--cache", required=True); s.add_argument("--folds", required=True)
    s.add_argument("--participants", required=True)
    s.add_argument("--cells", required=True, help="comma list of PIPELINExEDGE, e.g. P6xhybrid")
    s.add_argument("--which", default="all", help="all | none | comma list of ablation names")
    s.add_argument("--n-perm", type=int, default=0); s.add_argument("--perm-start", type=int, default=0)
    s.add_argument("--out", default="results/ablations.jsonl"); s.add_argument("--max-hours", type=float, default=10.5)
    s.add_argument("--device", default="cpu"); s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("report"); s.add_argument("--reports", default="reports"); s.add_argument("--cache", required=True)
    s.add_argument("--preproc", nargs="*"); s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=2000); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("select-cells"); s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--top", type=int, default=2); s.set_defaults(fn=cmd_select_cells)
    s = sub.add_parser("folds"); s.add_argument("--participants", required=True)
    s.add_argument("--seeds", default=None, help="comma list; default = config cv.seeds")
    s.add_argument("--out", default="splits/folds_ds004504.json"); s.set_defaults(fn=cmd_folds)
    s = sub.add_parser("demographics"); s.add_argument("--participants", required=True)
    s.add_argument("--folds", required=True); s.add_argument("--out", default="results/demographics.csv")
    s.set_defaults(fn=cmd_demographics)
    s = sub.add_parser("smoke"); s.add_argument("--workdir", default="smoke"); s.set_defaults(fn=cmd_smoke)
    for name, fn in (("bench", cmd_bench), ("grid", cmd_grid)):
        s = sub.add_parser(name)
        s.add_argument("--cache", required=True); s.add_argument("--folds", required=True)
        s.add_argument("--device", default="cpu"); s.add_argument("--max-hours", type=float, default=None)
        if name == "bench":
            s.add_argument("--pipeline", default="P1"); s.add_argument("--edge", default="hybrid")
            s.add_argument("--n-runs", type=int, default=175); s.add_argument("--out", default="results/bench.jsonl")
        else:
            s.add_argument("--pipelines", default="P1,P2,P3,P4,P5,P6,P7")
            s.add_argument("--edges", default="spatial,functional,hybrid")
            s.add_argument("--tag", default="main"); s.add_argument("--out", default="results/runs.jsonl")
            s.add_argument("--set", nargs="*", help="config overrides key=value")
            s.add_argument("--refit-from", nargs="*", help="main-grid runs.jsonl: refit on train+val")
        s.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if getattr(a, "max_hours", "absent") is None:
        a.max_hours = cfg["budget"]["max_hours"]
    a.fn(a, cfg)


if __name__ == "__main__":
    main()
