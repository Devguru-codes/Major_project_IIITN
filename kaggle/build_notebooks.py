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
