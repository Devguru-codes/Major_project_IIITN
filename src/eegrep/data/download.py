"""Download an OpenNeuro dataset from the public S3 bucket over plain HTTPS.

No AWS CLI or credentials needed. Only raw BIDS files are fetched (the
authors' `derivatives/` are skipped: the study applies its own preamble).
Resumable: files whose local size matches the listing are skipped. A
manifest with sizes and SHA-256 is written for the reproducibility record.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

S3_ROOT = "https://s3.amazonaws.com"
_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(bucket: str, prefix: str) -> list[tuple[str, int]]:
    objects, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = f"{S3_ROOT}/{bucket}?{urllib.parse.urlencode(query)}"
        root = ET.fromstring(_fetch(url))
        for c in root.findall("s:Contents", _NS):
            objects.append((c.find("s:Key", _NS).text, int(c.find("s:Size", _NS).text)))
        nxt = root.find("s:NextContinuationToken", _NS)
        if nxt is None:
            return objects
        token = nxt.text


def select_raw(objects: list[tuple[str, int]], dataset: str) -> list[tuple[str, int]]:
    """Keep top-level metadata and sub-*/ files; drop derivatives/ and git internals."""
    keep = []
    for key, size in objects:
        rel = key[len(dataset) + 1:]
        if rel.startswith("derivatives/") or rel.startswith(".git") or not rel:
            continue
        keep.append((key, size))
    return keep


def download_dataset(dataset: str, dest: str | Path, bucket: str = "openneuro.org",
                     subjects: list[str] | None = None, retries: int = 3) -> Path:
    dest = Path(dest)
    objects = select_raw(list_objects(bucket, f"{dataset}/"), dataset)
    if subjects is not None:
        wanted = set(subjects)
        objects = [(k, s) for k, s in objects if "/sub-" not in k or k.split("/")[1] in wanted]
    manifest = []
    for i, (key, size) in enumerate(objects, 1):
        out = dest / key[len(dataset) + 1:]
        if not (out.exists() and out.stat().st_size == size):
            out.parent.mkdir(parents=True, exist_ok=True)
            _download(f"{S3_ROOT}/{bucket}/{urllib.parse.quote(key)}", out, size, retries)
        manifest.append({"path": str(out.relative_to(dest)), "size": size, "sha256": _sha256(out)})
        if i % 25 == 0 or i == len(objects):
            print(f"[download] {i}/{len(objects)} files", flush=True)
    (dest / "download_manifest.json").write_text(json.dumps(
        {"dataset": dataset, "bucket": bucket, "version": _version(dest), "files": manifest}, indent=1))
    return dest


def _version(dest: Path) -> str:
    changes = dest / "CHANGES"
    return changes.read_text(encoding="utf-8").split()[0] if changes.exists() else "unknown"


def _fetch(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _download(url: str, out: Path, size: int, retries: int) -> None:
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {out.name}: {tmp.stat().st_size} != {size}")
            tmp.replace(out)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
