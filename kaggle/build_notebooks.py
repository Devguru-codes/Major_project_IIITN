"""Generate the Kaggle kernel folders (notebook + kernel-metadata.json).

Every notebook shares one setup cell (clone branch, pip install) so all kernels
run identical code. Re-run after editing, then `kaggle kernels push -p <folder>`.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OWNER = "dev123123456"            # account behind ~/.kaggle/access_token
N_SHARDS = 3
GRID_SHARDS = ["P1,P5", "P2,P3,P4", "P6,P7"]   # P5 (1280-dim) is the heaviest; balance by cost
# (grid-rank index of the cell, ablations, n permutations, first permutation) — ≤ 5 kernels (Kaggle CPU limit)
ABLATION_SHARDS = [
    (0, "A1_no_graph,A2_random_graph,A3_k3,A3_k6,A3_full,A4_binary,A5_depth1", 0, 0),
    (0, "A5_depth3,A6_gat,A9_unweighted,A12_residualized,RQ7_plus_P7,RQ7_pswe_edge,A10_window_split", 0, 0),
    (1, "all", 0, 0),
    (0, "none", 100, 0),
    (0, "none", 100, 100),
]

SETUP = r"""REPO = 'https://github.com/Devguru-codes/Major_project_IIITN.git'
BRANCH = 'feat/implementation'
WORK = '/kaggle/working'
SRC = '/tmp/eegrep-src'   # code checkout stays out of the saved output

import subprocess, sys, os, json, platform, shutil

def sh(cmd, log=None):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-20000:], r.stderr[-5000:], sep='\n', flush=True)
    if log:
        open(log, 'w').write(r.stdout + '\n' + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}')
    return r.stdout

shutil.rmtree(SRC, ignore_errors=True)
sh(f'git clone --depth 1 -b {BRANCH} {REPO} {SRC}')
SHA = sh(f'git -C {SRC} rev-parse --short HEAD').strip()
sh(f'pip install -q -e "{SRC}[{EXTRAS}]"', log=f'{WORK}/pip_install.log')
PY = f'cd {SRC} && {sys.executable} -m eegrep'
print('git sha', SHA)"""

ENV = r"""import torch
sh(f'{sys.executable} -m pip freeze', log=f'{WORK}/pip_freeze.txt')
env = {'git_sha': SHA, 'python': platform.python_version(), 'torch': torch.__version__,
       'cuda': torch.cuda.is_available(), 'cpu_count': os.cpu_count(),
       'kaggle_image': os.environ.get('KAGGLE_DOCKER_IMAGE', 'unknown')}
json.dump(env, open(f'{WORK}/environment.json', 'w'), indent=1)
print(env)"""

PARTICIPANTS_URL = "https://s3.amazonaws.com/openneuro.org/ds004504/participants.tsv"


def notebook(title: str, intro: str, cells: list[str], extras: str) -> dict:
    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}

    md = {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n\n", intro]}
    body = [code(SETUP.replace("{EXTRAS}", extras))] + [code(c) for c in cells] + [code(ENV)]
    return {"cells": [md] + body, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                              "name": "python3"},
                                               "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def write_kernel(slug: str, title: str, nb: dict, gpu: bool = False, internet: bool = True,
                 kernel_sources: list[str] | None = None) -> None:
    d = HERE / slug
    d.mkdir(exist_ok=True)
    fname = slug.replace("-", "_") + ".ipynb"
    (d / fname).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/eegrep-{slug}", "title": f"eegrep {title}", "code_file": fname,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu,
            "enable_tpu": False, "enable_internet": internet, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": kernel_sources or []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    write_kernel("nb00-setup-tests", "nb00 setup tests", notebook(
        "NB00 — setup + tests (CPU)",
        "Runs the full pytest suite and the synthetic end-to-end smoke run (no real data).",
        [f"sh(f'cd {{SRC}} && {{sys.executable}} -m pytest -q -p no:cacheprovider', log=f'{{WORK}}/pytest.txt')",
         "sh(f'{PY} smoke --workdir /tmp/eegrep-smoke', log=f'{WORK}/smoke.txt')\n"
         "shutil.copy('/tmp/eegrep-smoke/runs.jsonl', f'{WORK}/smoke_runs.jsonl')"],
        extras="dev,stats"))

    for k in range(N_SHARDS):
        write_kernel(f"nb01-preprocess-s{k}", f"nb01 preprocess s{k}", notebook(
            f"NB01 — download + fixed preamble + feature cache, shard {k}/{N_SHARDS} (CPU)",
            f"Subjects `sorted(participants)[{k}::{N_SHARDS}]`. Downloads only this shard's raw `.set` files "
            "from OpenNeuro S3, runs the fixed preamble (filter, notch, avg-ref, ICA+ICLabel, 128 Hz), "
            "then builds P1–P7 features, wPLI and PSWE outputs. Shards are merged in NB02.",
            [f"SHARD, N_SHARDS = {k}, {N_SHARDS}\n"
             "sh(f'{PY} download --dest /tmp/ds004504 --shard {SHARD} --n-shards {N_SHARDS}', log=f'{WORK}/download.log')\n"
             "shutil.copy('/tmp/ds004504/download_manifest.json', f'{WORK}/download_manifest_s{SHARD}.json')\n"
             "shutil.copy('/tmp/ds004504/participants.tsv', f'{WORK}/participants.tsv')",
             "sh(f'{PY} preprocess --bids /tmp/ds004504 --out {WORK}/preproc --shard {SHARD} "
             "--n-shards {N_SHARDS} --n-jobs 4', log=f'{WORK}/preprocess.log')",
             "sh(f'{PY} cache --preproc {WORK}/preproc --participants {WORK}/participants.tsv "
             "--out {WORK}/cache_s{SHARD}', log=f'{WORK}/cache.log')"],
            extras="preprocess"))

    write_kernel("nb02-merge-qc", "nb02 merge qc", notebook(
        "NB02 — merge shard caches + data-quality gates (CPU)",
        "Inputs: the outputs of the three NB01 shard kernels. Writes the single feature cache used by every "
        "training notebook and `qc_report.json` (window counts, durations, ICA removals, θ/α slowing check, "
        "PSWE burden per group).",
        ["import glob\n"
         "shards = sorted(glob.glob('/kaggle/input/**/cache_s*', recursive=True))\n"
         "logs = sorted(glob.glob('/kaggle/input/**/preproc/preprocess_log.jsonl', recursive=True))\n"
         "part = sorted(glob.glob('/kaggle/input/**/participants.tsv', recursive=True))[0]\n"
         f"print(shards, logs, part, sep='\\n')\n"
         f"assert len(shards) == {N_SHARDS} and len(logs) == {N_SHARDS}, 'missing NB01 shard outputs'\n"
         "shutil.copy(part, f'{WORK}/participants.tsv')",
         "sh(f'{PY} merge-caches --inputs {\" \".join(shards)} --out {WORK}/cache', log=f'{WORK}/merge.log')",
         "sh(f'{PY} qc --cache {WORK}/cache --logs {\" \".join(logs)} --out {WORK}/qc_report.json --no-fail')\n"
         "print(json.load(open(f'{WORK}/qc_report.json'))['gate_failures'])"],
        extras="dev"),
        internet=True,
        kernel_sources=[f"{OWNER}/eegrep-nb01-preprocess-s{k}" for k in range(N_SHARDS)])

    find_cache = ("import glob\n"
                  "CACHE = [d for d in glob.glob('/kaggle/input/**/cache', recursive=True) "
                  "if os.path.exists(f'{d}/index.npz')][0]\n"
                  "PART = sorted(glob.glob('/kaggle/input/**/participants.tsv', recursive=True))[0]\n"
                  "FOLDS = f'{SRC}/splits/folds_ds004504.json'\n"
                  "print(CACHE, PART, FOLDS)")
    nb02 = [f"{OWNER}/eegrep-nb02-merge-qc"]

    write_kernel("nb03b-pswe-stats", "nb03b pswe stats", notebook(
        "NB03b — PSWE (BBB-associated marker) statistics, RQ6 (CPU)",
        "Kruskal–Wallis + Dunn (Holm), NB2 regression of PSWE counts adjusted for age/sex (and for generic "
        "δ+θ slowing), PSWE-vs-slowing partial correlation, PSWE-vs-MMSE within patients (analysis only), "
        "per-channel topography.",
        [find_cache,
         "sh(f'{PY} pswe-stats --cache {CACHE} --participants {PART} --out {WORK}/rq6', log=f'{WORK}/rq6.log')"],
        extras="stats"), kernel_sources=nb02)

    write_kernel("nb03d-pswe-relative", "nb03d pswe relative", notebook(
        "NB03d — EXPLORATORY background-relative PSWE (CPU)",
        "Post hoc follow-up to RQ6: the fixed 6 Hz rule tracked background slowing (ρ = 0.88). Here a PSWE is "
        "transient slowing relative to each channel's own recording-median MPF (primary: ≥ 2 Hz drop for ≥ 5 s; "
        "sensitivity: 1.5/3 Hz drops, 2/3·MAD). Same statistics as RQ6, plus adjustment for background MPF.",
        ["sh(f'cd {SRC} && {sys.executable} -m pytest -q -p no:cacheprovider tests/test_pswe.py tests/test_stats.py', "
         "log=f'{WORK}/pytest_pswe.txt')",
         find_cache + "\n"
         "PRE = sorted(d for d in glob.glob('/kaggle/input/**/preproc', recursive=True) if glob.glob(f'{d}/sub-*.npz'))\n"
         f"print(PRE); assert len(PRE) == {N_SHARDS}",
         "sh(f'{PY} pswe-relative --preproc {\" \".join(PRE)} --cache {CACHE} --participants {PART} "
         "--out {WORK}/rq6_relative', log=f'{WORK}/rq6_relative.log')"],
        extras="stats"),
        kernel_sources=[f"{OWNER}/eegrep-nb01-preprocess-s{k}" for k in range(N_SHARDS)] + nb02)

    write_kernel("nb03c-classical-baselines", "nb03c classical baselines", notebook(
        "NB03c — classical non-graph baselines per representation (CPU)",
        "LR / RBF-SVM / RF on flattened window features of P1–P4, P6, P7 (P5 excluded: 24k dims), "
        "same frozen folds and weighting as the GCN, subject = mean window log-probability.",
        [find_cache,
         "sh(f'{PY} classical --cache {CACHE} --folds {FOLDS} --out {WORK}/classical.csv', log=f'{WORK}/classical.log')"],
        extras="stats"), kernel_sources=nb02)

    for k, pipes in enumerate(GRID_SHARDS):
        write_kernel(f"nb04-grid-s{k}", f"nb04 grid s{k}", notebook(
            f"NB04 — main 7×3 grid, shard {k} ({pipes}) (CPU)",
            f"Pipelines {pipes} × edges spatial, functional, hybrid × 5 repeats × 5 folds under the fixed GCN. "
            "Resumable; stops cleanly at the 11 h budget.",
            [find_cache,
             f"sh(f'{{PY}} bench --cache {{CACHE}} --folds {{FOLDS}} --pipeline {pipes.split(',')[-1]} --edge hybrid "
             f"--n-runs {len(pipes.split(',')) * 75} --max-hours 11 --out {{WORK}}/bench.jsonl', log=f'{{WORK}}/bench.log')",
             f"sh(f'{{PY}} grid --cache {{CACHE}} --folds {{FOLDS}} --pipelines {pipes} "
             f"--edges spatial,functional,hybrid --tag main --out {{WORK}}/runs.jsonl --max-hours 10.5', "
             f"log=f'{{WORK}}/grid.log')"],
            extras="dev"), kernel_sources=nb02)

    grid_sources = [f"{OWNER}/eegrep-nb04-grid-s{k}" for k in range(len(GRID_SHARDS))]
    select = ("RUNS = sorted(glob.glob('/kaggle/input/**/runs.jsonl', recursive=True))\n"
              f"print(RUNS); assert len(RUNS) == {len(GRID_SHARDS)}, 'missing grid shard outputs'\n"
              "CELLS = sh(f'{PY} select-cells --runs {\" \".join(RUNS)} --top 2').strip().splitlines()[-1].split(',')\n"
              "json.dump(CELLS, open(f'{WORK}/cells.json', 'w')); print('ablating', CELLS)")
    for k, (cell_idx, which, n_perm, start) in enumerate(ABLATION_SHARDS):
        write_kernel(f"nb05-ablations-s{k}", f"nb05 ablations s{k}", notebook(
            f"NB05 — ablations / permutation null, shard {k} (CPU)",
            f"Cell = main-grid rank {cell_idx + 1} (chosen on Kaggle from the NB04 outputs). Ablations: `{which}`. "
            f"Permutations: {n_perm} starting at {start}. Same frozen folds as the grid, so runs are paired.",
            [find_cache + "\n" + select,
             f"sh(f'{{PY}} ablate --cache {{CACHE}} --folds {{FOLDS}} --participants {{PART}} --cells {{CELLS[{cell_idx}]}} "
             f"--which {which} --n-perm {n_perm} --perm-start {start} --out {{WORK}}/ablations.jsonl --max-hours 10.5', "
             f"log=f'{{WORK}}/ablate.log')"],
            extras="dev"), kernel_sources=nb02 + grid_sources)

    write_kernel("nb07-stats-figures", "nb07 stats figures", notebook(
        "NB07 — statistics, tables and figures (CPU)",
        "Builds every journal table (LaTeX + CSV) and figure (PDF + 300-dpi PNG) from the raw runs committed under "
        "`reports/` plus the NB02 cache and NB01 preprocessed recordings. No hand-typed numbers.",
        ["sh(f'cd {SRC} && {sys.executable} -m pytest -q -p no:cacheprovider tests/test_stats.py "
         "tests/test_stats_ablation.py', log=f'{WORK}/pytest_stats.txt')",
         find_cache + "\n"
         "PRE = sorted(d for d in glob.glob('/kaggle/input/**/preproc', recursive=True) if glob.glob(f'{d}/sub-*.npz'))\n"
         "print(PRE)",
         "sh(f'{PY} report --reports {SRC}/reports --cache {CACHE} --preproc {\" \".join(PRE)} --out {WORK}/report', "
         "log=f'{WORK}/report.log')\n"
         "print(open(f'{WORK}/report/summary.json').read())"],
        extras="dev,stats"),
        kernel_sources=nb02 + [f"{OWNER}/eegrep-nb01-preprocess-s0"])

    # ---- must-fix re-runs after internal review (see IMPLEMENTATION_PLAN.md log) ----
    write_kernel("nb03f-tests-confirm-folds", "nb03f tests confirm folds", notebook(
        "NB03f — full tests + frozen confirmation folds (seeds 5–9) (CPU)",
        "Runs the full test suite, then generates the independent confirmation fold assignment used to "
        "re-evaluate the grid, ablations and permutation null free of selection bias. Committed before any "
        "confirmation run.",
        [f"sh(f'cd {{SRC}} && {{sys.executable}} -m pytest -q -p no:cacheprovider', log=f'{{WORK}}/pytest.txt')",
         f"sh(f'curl -sSf -o {{WORK}}/participants.tsv {PARTICIPANTS_URL}')\n"
         "sh(f'{PY} folds --participants {WORK}/participants.tsv --seeds 5,6,7,8,9 "
         "--out {WORK}/folds_ds004504_confirm.json')"],
        extras="dev,stats"))

    confirm = "{{SRC}}/splits/folds_ds004504_confirm.json"     # doubled braces survive .format below
    refit_src = "{{SRC}}/reports/grid/runs_s*.jsonl"
    job = {
        "grid_confirm": "sh(f'{{PY}} grid --cache {{CACHE}} --folds " + confirm + " --pipelines {p} "
                        "--edges spatial,functional,hybrid --tag confirm --out {{WORK}}/confirm_runs.jsonl "
                        "--max-hours 5', log=f'{{WORK}}/confirm_{n}.log')",
        "refit": "sh(f'{{PY}} grid --cache {{CACHE}} --folds {{FOLDS}} --pipelines {p} --edges spatial,functional,hybrid "
                 "--tag refit --refit-from " + refit_src + " --out {{WORK}}/refit_runs.jsonl --max-hours 5', "
                 "log=f'{{WORK}}/refit_{n}.log')",
        "ablate": "sh(f'{{PY}} ablate --cache {{CACHE}} --folds " + confirm + " --participants {{PART}} --cells {p} "
                  "--which all --out {{WORK}}/ablations_confirm.jsonl --max-hours 6', log=f'{{WORK}}/ablate_{n}.log')",
        "perm": "sh(f'{{PY}} ablate --cache {{CACHE}} --folds " + confirm + " --participants {{PART}} --cells {p} "
                "--which none --n-perm 100 --perm-start 1000 --out {{WORK}}/perm_confirm.jsonl --max-hours 5', "
                "log=f'{{WORK}}/perm_{n}.log')",
        "classical_train": "sh(f'{{PY}} classical --cache {{CACHE}} --folds {{FOLDS}} --fit-on train "
                           "--out {{WORK}}/classical_train.csv', log=f'{{WORK}}/classical_{n}.log')",
    }
    RERUN_SHARDS = [
        [("grid_confirm", "P1,P2,P5"), ("perm", "P3xspatial")],
        [("grid_confirm", "P3,P4,P6,P7"), ("refit", "P1,P2,P3")],
        [("ablate", "P3xspatial"), ("refit", "P4,P5")],
        [("ablate", "P3xhybrid"), ("refit", "P6,P7")],
        [("classical_train", "all")],
    ]
    for k, jobs in enumerate(RERUN_SHARDS):
        cells = [find_cache] + [job[j].format(p=p, n=i) for i, (j, p) in enumerate(jobs)]
        write_kernel(f"nb09-rerun-s{k}", f"nb09 rerun s{k}", notebook(
            f"NB09 — must-fix re-runs, shard {k} (CPU)",
            "Jobs: " + "; ".join(f"`{j}` {p}" for j, p in jobs) + ". Confirmation runs use fresh frozen folds "
            "(seeds 5–9); refits train the GCN on train+val for the early-stopped epoch count of the matching "
            "main-grid run; classical baselines are refit on the GCN's training subjects only.",
            cells, extras="dev"), kernel_sources=nb02)

    # ---- early-stopping sensitivity (selection folds, paired with the main grid) ----
    es_set = {"es_p50": "train.patience=50", "es_fixed100": "train.patience=null train.max_epochs=100"}
    es_test = ("sh(f'cd {SRC} && {sys.executable} -m pytest -q -p no:cacheprovider tests/test_model_train.py "
               "tests/test_stats_ablation.py', log=f'{WORK}/pytest_es.txt')")
    ES_SHARDS = [
        [("es_fixed100", "P5"), ("es_p50", "P7")],
        [("es_p50", "P5"), ("es_fixed100", "P1")],
        [("es_fixed100", "P2,P3"), ("es_p50", "P6")],
        [("es_fixed100", "P4,P6,P7")],
        [("es_p50", "P1,P2,P3,P4")],
    ]
    for k, jobs in enumerate(ES_SHARDS):
        cells = [es_test, find_cache] + [
            f"sh(f'{{PY}} grid --cache {{CACHE}} --folds {{FOLDS}} --pipelines {p} --edges spatial,functional,hybrid "
            f"--tag {tag} --set {es_set[tag]} --out {{WORK}}/es_runs.jsonl --max-hours 9', log=f'{{WORK}}/{tag}_{i}.log')"
            for i, (tag, p) in enumerate(jobs)]
        write_kernel(f"nb10-es-sensitivity-s{k}", f"nb10 es sensitivity s{k}", notebook(
            f"NB10 — early-stopping sensitivity, shard {k} (CPU)",
            "Jobs: " + "; ".join(f"`{t}` {p}" for t, p in jobs) + ". Same 25 selection splits as the main grid; "
            "only the stopping rule changes (patience 50, or no early stopping: 100 epochs on the training "
            "subjects, final weights).",
            cells, extras="dev"), kernel_sources=nb02)

    # ---- external validation on ds004584 (PD vs controls) ----
    ext = "EXT = f'{PY} --config configs/ds004584.yaml'\n"
    write_kernel("nb11a-external-prep", "nb11a external prep", notebook(
        "NB11a — external cohort ds004584: download, fixed preamble, features, PSWE, frozen folds (CPU)",
        "Same code path as NB01–NB03 with the `configs/ds004584.yaml` overlay (Pz reference restored, 60 Hz notch, "
        "eyes-open task, PD/Control labels, MoCA as the analysis-only cognitive score). The fold file is "
        "committed to the repository before any training run on this cohort (NB11b).",
        [f"sh(f'cd {{SRC}} && {{sys.executable}} -m pytest -q -p no:cacheprovider', log=f'{{WORK}}/pytest.txt')",
         ext + "sh(f'{EXT} download --dest /tmp/ds004584', log=f'{WORK}/download.log')\n"
         "shutil.copy('/tmp/ds004584/download_manifest.json', f'{WORK}/download_manifest.json')\n"
         "shutil.copy('/tmp/ds004584/participants.tsv', f'{WORK}/participants.tsv')",
         ext + "sh(f'{EXT} preprocess --bids /tmp/ds004584 --out {WORK}/preproc --n-jobs 4', log=f'{WORK}/preprocess.log')",
         ext + "sh(f'{EXT} cache --preproc {WORK}/preproc --participants {WORK}/participants.tsv --out {WORK}/cache', "
         "log=f'{WORK}/cache.log')\n"
         "sh(f'{EXT} qc --cache {WORK}/cache --logs {WORK}/preproc/preprocess_log.jsonl --out {WORK}/qc_report.json --no-fail')\n"
         "print(json.load(open(f'{WORK}/qc_report.json'))['gate_failures'])",
         ext + "sh(f'{EXT} folds --participants {WORK}/participants.tsv --out {WORK}/folds_ds004584.json')\n"
         "sh(f'{EXT} demographics --participants {WORK}/participants.tsv --folds {WORK}/folds_ds004584.json "
         "--out {WORK}/demographics.csv', log=f'{WORK}/demographics.txt')",
         ext + "sh(f'{EXT} pswe-stats --cache {WORK}/cache --participants {WORK}/participants.tsv --out {WORK}/rq6', "
         "log=f'{WORK}/pswe_stats.txt')\n"
         "sh(f'{EXT} pswe-relative --preproc {WORK}/preproc --cache {WORK}/cache --participants {WORK}/participants.tsv "
         "--out {WORK}/rq6_relative', log=f'{WORK}/pswe_relative.txt')"],
        extras="preprocess,dev,stats"))

    ext_folds = "{SRC}/splits/folds_ds004584.json"
    EXT_SHARDS = [
        [f"sh(f'{{EXT}} grid --cache {{CACHE}} --folds {ext_folds} --pipelines P1,P2,P3,P4 --tag main "
         "--out {WORK}/ext_runs.jsonl --max-hours 6', log=f'{WORK}/grid_0.log')",
         f"sh(f'{{EXT}} classical --cache {{CACHE}} --folds {ext_folds} --fit-on train --out {{WORK}}/cls_train.csv', "
         "log=f'{WORK}/classical_train.log')\n"
         f"sh(f'{{EXT}} classical --cache {{CACHE}} --folds {ext_folds} --fit-on trainval --out {{WORK}}/cls_tv.csv', "
         "log=f'{WORK}/classical_tv.log')\n"
         "import pandas as pd\n"
         "pd.concat([pd.read_csv(f'{WORK}/cls_train.csv'), pd.read_csv(f'{WORK}/cls_tv.csv')]).to_csv("
         "f'{WORK}/classical.csv', index=False)"],
        [f"sh(f'{{EXT}} grid --cache {{CACHE}} --folds {ext_folds} --pipelines P5,P6,P7 --tag main "
         "--out {WORK}/ext_runs.jsonl --max-hours 6', log=f'{WORK}/grid_1.log')",
         f"sh(f'{{EXT}} ablate --cache {{CACHE}} --folds {ext_folds} --participants {{PART}} --cells P3xspatial "
         "--which A10_window_split --n-perm 100 --perm-start 0 --out {WORK}/ext_ablations.jsonl --max-hours 6', "
         "log=f'{WORK}/ablate.log')"],
    ]
    for k, jobs in enumerate(EXT_SHARDS):
        write_kernel(f"nb11b-external-train-s{k}", f"nb11b external train s{k}", notebook(
            f"NB11b — external cohort ds004584: fixed GCN grid and controls, shard {k} (CPU)",
            "The unchanged 7 x 3 protocol as a two-class (PD vs control) problem on the frozen ds004584 folds; "
            "shard 0 adds the non-graph twins, shard 1 the leakage experiment and 100 label permutations for the "
            "configuration selected on ds004504 (P3 x spatial).",
            [f"sh(f'cd {{SRC}} && {{sys.executable}} -m pytest -q -p no:cacheprovider tests/test_external.py "
             "tests/test_model_train.py', log=f'{WORK}/pytest_ext.txt')",
             find_cache + "\n" + ext + f"assert os.path.exists(f'{ext_folds}'), 'commit the NB11a folds first'"] + jobs,
            extras="dev"), kernel_sources=[f"{OWNER}/eegrep-nb11a-external-prep"])

    # ---- ablations A7 (window length) and A8 (ICA off): new caches, confirmation folds, both reference cells ----
    confirm_folds = "{SRC}/splits/folds_ds004504_confirm.json"

    def train_cells(name, cache, out):
        return "\n".join(
            f"sh(f'{{C}} grid --cache {cache} --folds {confirm_folds} --pipelines P3 --edges {e} "
            f"--tag {name}@P3x{e} --out {{WORK}}/{out} --max-hours 5', log=f'{{WORK}}/{name}_{e}.log')"
            for e in ("spatial", "hybrid"))

    abl_test = ("sh(f'cd {SRC} && {sys.executable} -m pytest -q -p no:cacheprovider tests/test_external.py "
                "tests/test_model_train.py', log=f'{WORK}/pytest_abl.txt')")
    for name, conf in (("A7_window5", "ablation_a7_window5"), ("A7_window20", "ablation_a7_window20")):
        slug = name.lower().replace("_", "-")
        write_kernel(f"nb12-{slug}", f"nb12 {name.lower()}", notebook(
            f"NB12 — ablation {name}: re-windowed feature cache + reference cells on the confirmation folds (CPU)",
            "Re-windows the NB01 preprocessed recordings (only `preprocess.window_s` changes), rebuilds every "
            "feature and graph, then trains P3 x spatial and P3 x hybrid on the 25 confirmation splits, paired with "
            "their confirmation-grid runs.",
            [abl_test,
             "import glob\n"
             "PRE = sorted(d for d in glob.glob('/kaggle/input/**/preproc', recursive=True) if glob.glob(f'{d}/sub-*.npz'))\n"
             "PART = sorted(glob.glob('/kaggle/input/**/participants.tsv', recursive=True))[0]\n"
             f"C = f'{{PY}} --config configs/{conf}.yaml'\n"
             "print(PRE, PART); assert len(PRE) == 3",
             "for k, d in enumerate(PRE):\n"
             f"    sh(f'{{C}} cache --preproc {{d}} --participants {{PART}} --out /tmp/cache_s{{k}}', log=f'{{WORK}}/cache_s{{k}}.log')\n"
             "sh(f'{C} merge-caches --inputs /tmp/cache_s0 /tmp/cache_s1 /tmp/cache_s2 --out /tmp/cache')\n"
             f"sh(f'{{C}} qc --cache /tmp/cache --out {{WORK}}/qc_{name}.json --no-fail')",
             train_cells(name, "/tmp/cache", "ablations_confirm.jsonl")],
            extras="preprocess,dev"),
            kernel_sources=[f"{OWNER}/eegrep-nb01-preprocess-s{k}" for k in range(N_SHARDS)])

    write_kernel("nb12-a8-no-ica", "nb12 a8 no ica", notebook(
        "NB12 — ablation A8: preprocessing without ICA + reference cells on the confirmation folds (CPU)",
        "Re-downloads ds004504, runs the fixed preamble with ICA/ICLabel skipped (`preprocess.ica_method: none`), "
        "rebuilds the cache, trains P3 x spatial and P3 x hybrid on the 25 confirmation splits, and repeats the "
        "fixed-threshold PSWE statistics (eye movements are slow and could masquerade as slow-wave events).",
        [abl_test,
         "C = f'{PY} --config configs/ablation_a8_no_ica.yaml'\n"
         "sh(f'{C} download --dest /tmp/ds004504', log=f'{WORK}/download.log')\n"
         "shutil.copy('/tmp/ds004504/participants.tsv', f'{WORK}/participants.tsv')\n"
         "sh(f'{C} preprocess --bids /tmp/ds004504 --out /tmp/preproc --n-jobs 4', log=f'{WORK}/preprocess.log')\n"
         "shutil.copy('/tmp/preproc/preprocess_log.jsonl', f'{WORK}/preprocess_log.jsonl')",
         "sh(f'{C} cache --preproc /tmp/preproc --participants {WORK}/participants.tsv --out /tmp/cache', "
         "log=f'{WORK}/cache.log')\n"
         "sh(f'{C} qc --cache /tmp/cache --logs /tmp/preproc/preprocess_log.jsonl --out {WORK}/qc_A8_no_ica.json --no-fail')\n"
         "sh(f'{C} pswe-stats --cache /tmp/cache --participants {WORK}/participants.tsv --out {WORK}/rq6_no_ica', "
         "log=f'{WORK}/pswe_stats.txt')",
         train_cells("A8_no_ica", "/tmp/cache", "ablations_confirm.jsonl")],
        extras="preprocess,dev,stats"))

    # ---- primary inference: subject-level linear mixed model (crossed subject / split / model intercepts) ----
    write_kernel("nb13-mixed-model", "nb13 mixed model", notebook(
        "NB13 — subject-level mixed-effects analysis of the factorial (CPU)",
        "Fits p_true ~ representation x graph + true class + (1|subject) + (1|split) + (1|cell x split) on the "
        "committed runs of the selection folds, the confirmation folds and the external cohort (ds004584).",
        ["sh(f'cd {SRC} && {sys.executable} -m pytest -q -p no:cacheprovider tests/test_mixed.py', "
         "log=f'{WORK}/pytest_mixed.txt')",
         "sh(f'{PY} mixed --runs {SRC}/reports/grid/runs_s0.jsonl {SRC}/reports/grid/runs_s1.jsonl "
         "{SRC}/reports/grid/runs_s2.jsonl --tag main --out {WORK}/mixed --prefix selection', log=f'{WORK}/mixed_selection.log')",
         "sh(f'{PY} mixed --runs {SRC}/reports/confirm/confirm_runs_s0.jsonl {SRC}/reports/confirm/confirm_runs_s1.jsonl "
         "--tag confirm --out {WORK}/mixed --prefix confirmation', log=f'{WORK}/mixed_confirmation.log')",
         "sh(f'{PY} mixed --runs {SRC}/reports/external/ext_runs_s0.jsonl {SRC}/reports/external/ext_runs_s1.jsonl "
         "--tag main --out {WORK}/mixed --prefix external', log=f'{WORK}/mixed_external.log')",
         "import pandas as pd\n"
         "sys.path.insert(0, f'{SRC}/src')\n"
         "from eegrep import figures as F\n"
         "F.fig_mixed({'selection (seeds 0-4)': pd.read_csv(f'{WORK}/mixed/selection_marginal.csv'),\n"
         "             'confirmation (seeds 5-9)': pd.read_csv(f'{WORK}/mixed/confirmation_marginal.csv')},\n"
         "            f'{WORK}/mixed/figures')"],
        extras="dev,stats"))

    write_kernel("nb03a-folds-demographics", "nb03a folds demographics", notebook(
        "NB03a — frozen folds + demographics confound check (CPU)",
        "Writes the frozen subject-level fold assignment (5 repeats x 5 folds) and runs the "
        "age+sex-only baseline. If it is far above chance (macro-F1 ~0.33), stop and discuss confounding.",
        [f"sh(f'curl -sSf -o {{WORK}}/participants.tsv {PARTICIPANTS_URL}')\n"
         "sh(f'{PY} folds --participants {WORK}/participants.tsv --out {WORK}/folds_ds004504.json')",
         "sh(f'{PY} demographics --participants {WORK}/participants.tsv --folds {WORK}/folds_ds004504.json "
         "--out {WORK}/demographics.csv', log=f'{WORK}/demographics.txt')"],
        extras="dev"))


if __name__ == "__main__":
    main()
