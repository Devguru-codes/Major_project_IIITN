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
    """A config may name a `base` file (relative to itself); its keys are deep-merged onto the base, so a
    dataset overlay (e.g. configs/ds004584.yaml) changes only what differs between datasets."""
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return apply_overrides(cfg, overrides or {})


def _deep_merge(base: dict, over: dict) -> dict:
    """section.key values in the overlay replace the base's wholesale (so mappings such as
    group_to_label are swapped, not merged); unknown sections or keys are an error."""
    out = copy.deepcopy(base)
    for section, keys in over.items():
        if section not in out:
            raise KeyError(f"overlay section not in base config: {section}")
        for k, v in keys.items():
            if k not in out[section]:
                raise KeyError(f"overlay key not in base config: {section}.{k}")
            out[section][k] = copy.deepcopy(v)
    return out


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
