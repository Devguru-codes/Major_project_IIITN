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
        extras="dev"))

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
