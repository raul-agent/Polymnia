#!/usr/bin/env python3
"""Compatibility adapter between `parole` output and `hefesto.py` expectations.

Why this exists
---------------
The upstream POLYMNIA scripts disagree about where prosody data lives and how
its frame column is spelled:

* `ecos.py` calls `parole.sh`, which writes
      dataset/{n}_persons/parquet_speech_analysis/{id}/{id}_prosody.parquet
  with a frame column named `frame_id`.
* `hefesto.py` looks for
      dataset/{n}_persons/parquet_speech_analysis/{id}.parquet
  and renames a column called `Frame` to `frame`.

So a full pipeline run always ends with
`[WARNING] Prosody file not found:` and an empty/partial DuckDB.

This adapter materialises the flat layout that `hefesto.py` expects, derived
from what `parole` actually produced, without touching the original files and
without editing any upstream script. It is idempotent and reversible: the
generated files are listed in a manifest you can delete.

Usage
-----
    polymnia merge --input_folder out --n_persons 1 --db_name out/data.duckdb
        # runs this adapter automatically, then hefesto.py

    python3 tools/polymnia_prosody_compat.py --input_folder out --n_persons 1 [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, List

import pandas as pd

FRAME_COLUMN_CANDIDATES = ("Frame", "frame_id", "frame")
MANIFEST_NAME = ".prosody_compat_manifest.json"


def discover_prosody_files(speech_dir: str) -> Iterable[tuple[str, str]]:
    """Yield (clip_id, path) for prosody parquets parole produced.

    Handles both the nested parole layout ({id}/{id}_prosody.parquet) and an
    already-flat layout ({id}.parquet).
    """
    if not os.path.isdir(speech_dir):
        return []
    found: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(speech_dir)):
        full = os.path.join(speech_dir, entry)
        if os.path.isdir(full):
            inner = [f for f in sorted(os.listdir(full)) if f.endswith(".parquet")]
            if not inner:
                continue
            # Prefer the *_prosody.parquet file if several exist.
            preferred = [f for f in inner if f.endswith("_prosody.parquet")] or inner
            for f in preferred:
                found.append((entry, os.path.join(full, f)))
        elif entry.endswith(".parquet"):
            found.append((entry[: -len(".parquet")], full))
    return found


def normalize(src: str, dst: str) -> pd.DataFrame:
    """Read a prosody parquet and write the flat/`Frame` form hefesto expects."""
    df = pd.read_parquet(src)
    frame_col = next((c for c in FRAME_COLUMN_CANDIDATES if c in df.columns), None)
    if frame_col is None:
        raise ValueError(
            f"{src}: no frame column found (looked for {FRAME_COLUMN_CANDIDATES})"
        )
    if frame_col != "Frame":
        df = df.rename(columns={frame_col: "Frame"})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df.to_parquet(dst, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input_folder", required=True, help="pipeline output root")
    ap.add_argument("--n_persons", type=int, default=1)
    ap.add_argument(
        "--force",
        action="store_true",
        help="rewrite generated files even if they already exist",
    )
    args = ap.parse_args()

    n_persons = f"{args.n_persons}_persons"
    speech_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_speech_analysis")
    pose_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")

    if not os.path.isdir(speech_dir):
        print(f"[compat] no prosody folder at {speech_dir}; nothing to do")
        return 0
    if not os.path.isdir(pose_dir):
        print(f"[compat] no pose parquet folder at {pose_dir}; nothing to do")
        return 0

    pose_ids = {
        f[: -len(".parquet")] for f in os.listdir(pose_dir) if f.endswith(".parquet")
    }
    manifest_path = os.path.join(speech_dir, MANIFEST_NAME)
    manifest: List[str] = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    created = skipped = missing = 0
    for clip_id, src in discover_prosody_files(speech_dir):
        flat = os.path.join(speech_dir, f"{clip_id}.parquet")
        if os.path.abspath(src) == os.path.abspath(flat):
            # Already in the flat layout hefesto reads; nothing to adapt.
            skipped += 1
            continue
        if clip_id not in pose_ids:
            missing += 1
            print(f"[compat] {clip_id}: no matching pose parquet, skipped")
            continue
        if os.path.exists(flat) and not args.force and clip_id in manifest:
            skipped += 1
            continue
        # Never overwrite a file this adapter did not create.
        if os.path.exists(flat) and clip_id not in manifest and not args.force:
            print(f"[compat] {flat} already exists and was not generated here; skipped")
            skipped += 1
            continue
        try:
            df = normalize(src, flat)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"[compat] {clip_id}: FAILED ({exc})")
            continue
        manifest.append(clip_id)
        created += 1
        print(f"[compat] {clip_id}: {os.path.relpath(src, speech_dir)} -> {clip_id}.parquet ({len(df)} rows)")

    with open(manifest_path, "w") as fh:
        json.dump(sorted(set(manifest)), fh, indent=2)

    print(
        f"[compat] created={created} skipped={skipped} "
        f"without_pose_match={missing} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
