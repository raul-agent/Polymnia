#!/usr/bin/env python3
"""Build a synthetic-but-valid 50-line NewsScape tab file for pipeline testing.

The clip ids are real Democracy Now! programmes that gallo.case.edu actually
serves; only the in/out points are invented for the smoke test. The column
convention follows download_clips.py:

    name.txt <TAB> clip_id <TAB> start_sec <TAB> start_ms <TAB> end_sec <TAB> end_ms

Pre-2017 ids carry centiseconds (download_clips.py divides them by 100), and the
name keeps the NewsScape `..._US_WWW_Democracy_Now` shape plus the integer
second range that dfMaker parses out of it.
"""

from __future__ import annotations

import argparse
import random

# Days/hours confirmed to answer 307 (a cached snippet) from the snippet CGI.
VALID_CLIPS = [
    "2004-03-01_1400_US_WWW_Democracy_Now",
    "2004-03-02_1400_US_WWW_Democracy_Now",
    "2004-03-03_1400_US_WWW_Democracy_Now",
    "2004-03-04_1400_US_WWW_Democracy_Now",
    "2004-03-05_1400_US_WWW_Democracy_Now",
    "2004-03-08_1400_US_WWW_Democracy_Now",
    "2004-03-09_1400_US_WWW_Democracy_Now",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", type=int, default=50)
    ap.add_argument("--min-len", type=float, default=12.0)
    ap.add_argument("--max-len", type=float, default=22.0)
    ap.add_argument("--start-min", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20040301)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.out, "w") as fh:
        for _ in range(args.lines):
            clip = rng.choice(VALID_CLIPS)
            start_sec = rng.randrange(args.start_min, 3000)
            # centiseconds (0-99): pre-2017 convention, /100 in download_clips.py
            start_cs = rng.randrange(0, 100)
            dur = rng.uniform(args.min_len, args.max_len)
            end_abs = start_sec + start_cs / 100 + dur
            end_sec = int(end_abs)
            end_cs = round((end_abs - end_sec) * 100)
            if end_cs >= 100:  # rounding can spill into the next second
                end_sec += 1
                end_cs = 0
            # NewsScape names carry the integer second range dfMaker extracts.
            name = f"{clip}_{start_sec}-{end_sec}_before.txt"
            fh.write("\t".join([name, clip, str(start_sec), f"{start_cs:02d}",
                                str(end_sec), f"{end_cs:02d}"]) + "\n")
    print(f"{args.lines} filas -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
