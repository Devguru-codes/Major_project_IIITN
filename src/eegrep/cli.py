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


def cmd_download(a, cfg):
    from .data.download import download_dataset

    download_dataset(cfg["dataset"]["name"], a.dest, cfg["dataset"]["s3_bucket"])


def cmd_folds(a, cfg):
    from .data.participants import load_participants
    from .splits import file_sha256, make_folds, save_folds

    df = load_participants(a.participants, cfg["dataset"]["group_to_label"])
    cv = cfg["cv"]
    folds = make_folds(df["subject"], df["label"], cv["seeds"], cv["n_folds"], cv["val_frac"],
                       meta={"dataset": cfg["dataset"]["cite_doi"],
                             "participants_sha256": file_sha256(a.participants)})
    save_folds(folds, a.out)
    print(f"[folds] {len(df)} subjects, {len(cv['seeds'])} repeats x {cv['n_folds']} folds -> {a.out}")


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
    df = load_participants(work / "participants.tsv", cfg["dataset"]["group_to_label"])
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
    for p in a.pipelines.split(","):
        for e in a.edges.split(","):
            run_cell(cache, folds, cfg, store, pipeline=p, edge=e, tag=a.tag, deadline=deadline,
                     device=a.device, overrides=_overrides(a.set))
    print(json.dumps({"runs_done": len(store.done)}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eegrep")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download"); s.add_argument("--dest", required=True); s.set_defaults(fn=cmd_download)
    s = sub.add_parser("folds"); s.add_argument("--participants", required=True)
    s.add_argument("--out", default="splits/folds_ds004504.json"); s.set_defaults(fn=cmd_folds)
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
        s.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if getattr(a, "max_hours", "absent") is None:
        a.max_hours = cfg["budget"]["max_hours"]
    a.fn(a, cfg)


if __name__ == "__main__":
    main()
