"""Config loading, overrides and hashing.

The repo is installed editable (`pip install -e .`), so configs/ is found
relative to the source tree.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or {})


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides, e.g. {"graph.knn_k": 6}. Unknown keys are an error."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[leaf] = value
    return cfg


def config_hash(cfg: dict, sections=("preprocess", "bands", "features", "pswe", "graph", "model", "train")) -> str:
    """Short hash of everything that changes a run's result (not CV layout or budget)."""
    payload = json.dumps({s: cfg[s] for s in sections}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]
