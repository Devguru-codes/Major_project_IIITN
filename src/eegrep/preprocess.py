"""The fixed preamble — one identical code path for every condition.

band-pass 0.5–45 Hz -> 50 Hz notch -> average reference -> ICA (extended infomax,
fit on a 1 Hz high-passed copy) + ICLabel, drop eye/muscle components with
p >= 0.8 -> resample 128 Hz. Output is continuous µV data; windowing happens in
cache.build_cache.

ICA is fit per recording, without labels or other subjects' data, so it cannot
leak across CV folds (cross-subject transforms are fit in train.py on training
subjects only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ICA_RANDOM_STATE = 97


def preprocess_recording(set_path: str | Path, cfg: dict) -> tuple[np.ndarray, float, dict]:
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    pc, ds = cfg["preprocess"], cfg["dataset"]
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
    raw.pick(ds["channels"])
    raw.reorder_channels(ds["channels"])
    raw.rename_channels(ds["channel_rename"])
    raw.set_montage("standard_1020")
    sfreq_in, duration = raw.info["sfreq"], raw.n_times / raw.info["sfreq"]

    raw.filter(pc["l_freq"], pc["h_freq"], verbose="error")
    raw.notch_filter(pc["notch"], verbose="error")
    raw.set_eeg_reference(pc["reference"], verbose="error")

    ica_raw = raw.copy().filter(pc["ica_hp"], None, verbose="error")
    n_comp = len(ds["channels"]) - 1                     # average reference removes one rank
    ica = ICA(n_components=n_comp, method="infomax", fit_params={"extended": True},
              random_state=ICA_RANDOM_STATE, max_iter="auto")
    ica.fit(ica_raw, decim=max(1, int(sfreq_in // 125)), verbose="error")
    ic = label_components(ica_raw, ica, method="iclabel")
    labels, proba = ic["labels"], np.asarray(ic["y_pred_proba"])
    drop = [i for i, (lab, p) in enumerate(zip(labels, proba))
            if lab in pc["iclabel_drop"] and p >= pc["iclabel_threshold"]]
    ica.apply(raw, exclude=drop, verbose="error")

    raw.resample(pc["sfreq"], verbose="error")
    data = (raw.get_data() * 1e6).astype(np.float32)     # volts -> µV
    log = {"sfreq_in": sfreq_in, "duration_s": duration, "n_components": n_comp,
           "excluded": drop, "excluded_labels": [labels[i] for i in drop],
           "ic_labels": list(labels), "ic_proba": [round(float(p), 3) for p in proba]}
    return data, float(pc["sfreq"]), log


def preprocess_dataset(bids_root: str | Path, subjects: list[str], cfg: dict, out_dir: str | Path,
                       n_jobs: int = 1) -> None:
    """Write <out>/<subject>.npz (data µV float32, sfreq) and append to <out>/preprocess_log.jsonl."""
    from joblib import Parallel, delayed

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in subjects if not (out / f"{s}.npz").exists()]

    def one(subject):
        set_path = Path(bids_root) / subject / "eeg" / f"{subject}_task-eyesclosed_eeg.set"
        data, sfreq, log = preprocess_recording(set_path, cfg)
        np.savez(out / f"{subject}.npz", data=data, sfreq=sfreq)
        return {"subject": subject, **log}

    logs = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(s) for s in todo)
    with open(out / "preprocess_log.jsonl", "a", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")


def iter_preprocessed(pre_dir: str | Path, subjects: list[str]):
    for s in subjects:
        z = np.load(Path(pre_dir) / f"{s}.npz")
        yield s, z["data"], float(z["sfreq"])
