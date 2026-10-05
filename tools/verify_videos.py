#!/usr/bin/env python3
"""Verify downloaded NewsScape clips are actually playable video.

download_clips.py only checks that curl exited 0 and the file is non-empty. The
NewsScape snippet CGI answers HTTP 500 with an HTML body, and clips can also come
back truncated, so a folder full of `.mp4` files can contain things OpenPose and
WhisperX will silently fail on later in the pipeline.

This walks a folder, probes each file with ffprobe and reports:
  OK      decodable video stream with frames
  EMPTY   0 bytes
  NOTMP4  HTML/text where video should be
  NOVIDEO decodable container but no usable video stream
  BAD     ffprobe failed

Usage:
    verify_videos.py VIDEO_DIR [--delete-bad] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

def classify(path: str) -> tuple[str, str]:
    size = os.path.getsize(path)
    if size == 0:
        return "EMPTY", "0 bytes"
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:5] in (b"<!DOC", b"<html", b"<?xml"):
        return "NOTMP4", f"HTML de error del CGI ({size} B)"

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-count_frames",
                "-select_streams", "v:0", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "BAD", "ffprobe timeout"
    if out.returncode != 0:
        return "BAD", (out.stderr.strip().splitlines() or ["ffprobe error"])[-1][:120]
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return "BAD", "ffprobe salida ilegible"
    streams = data.get("streams") or []
    if not streams:
        return "NOVIDEO", "sin stream de video"
    st = streams[0]
    frames = st.get("nb_read_frames") or st.get("nb_frames")
    dur = st.get("duration")
    try:
        nframes = int(frames) if frames is not None else 0
    except (TypeError, ValueError):
        nframes = 0
    if nframes == 0:
        return "NOVIDEO", "stream de video sin frames"
    return "OK", f"{nframes} frames, {dur or '?'} s, {size} B"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir")
    ap.add_argument("--delete-bad", action="store_true",
                    help="borra los ficheros que no sean video util (EMPTY/NOTMP4/NOVIDEO/BAD)")
    ap.add_argument("--quiet", action="store_true", help="solo el resumen")
    args = ap.parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"no existe el directorio: {args.video_dir}")
        return 2

    files = [
        os.path.join(root, f)
        for root, _, fs in os.walk(args.video_dir)
        for f in sorted(fs) if f.lower().endswith(".mp4")
    ]
    if not files:
        print(f"sin .mp4 en {args.video_dir}")
        return 0

    counts: dict[str, int] = {}
    bad_paths: list[str] = []
    for p in files:
        status, detail = classify(p)
        counts[status] = counts.get(status, 0) + 1
        if status != "OK":
            bad_paths.append(p)
        if not args.quiet or status != "OK":
            print(f"[{status:7s}] {os.path.basename(p)} - {detail}")

    deleted = 0
    if args.delete_bad and bad_paths:
        for p in bad_paths:
            os.remove(p)
            deleted += 1
        print(f"borrados {deleted} ficheros no utilizables")

    ok = counts.get("OK", 0)
    print(f"\ntotal={len(files)} OK={ok} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "OK"))
    if deleted:
        print("los borrados se pueden volver a descargar: relanza download con el mismo --txt_file")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
