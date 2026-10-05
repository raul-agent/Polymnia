#!/usr/bin/env python3
"""Dry-run checker for POLYMNIA clip tab files (download_clips.py input).

Reproduces the timestamp logic of `download_clips.py` without touching the
network, and cross-checks it against the range encoded in the clip filename.

Conventions (this is the subtle part):
  * filenames from 2017 onwards have a field named `centiseconds` for backward
    compatibility, but the value really holds MILLISECONDS -> divide by 1000.
  * pre-2017 rows are genuine centiseconds -> divide by 100.

Most clip names look like
    2017-07-21_0000_US_KABC_Eyewitness_News_5PM_356.505_361.531_before
which already contains the un-offset start/end. That makes the filename an
independent source of truth: if the numbers computed from the tab columns do not
match it, the divisor or the tab itself is wrong.

Usage:
    check_tab.py TAB_FILE [--offset 2] [--show N]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# ..._356.505_361.531_before / ..._239-242_far : range embedded in the name
NAME_RANGE = re.compile(r"_(\d+)\.(\d+)_(\d+)\.(\d+)(?:_[A-Za-z]+)?$")
# older NewsScape style: _239-242_far (whole seconds only)
NAME_RANGE_S = re.compile(r"_(\d+)-(\d+)(?:_[A-Za-z]+)?$")


def divisor_for(name: str) -> int:
    """2017+ encodes milliseconds under the legacy `centiseconds` label."""
    return 1000 if os.path.basename(name).startswith("2017") else 100


def name_range(name: str) -> tuple[float, float, float] | None:
    """Return (start, end, tolerance) encoded in the clip name, if any.

    The tolerance reflects how much precision the name itself carries: fractional
    names pin the value to ~10ms, whole-second names only constrain the integer
    second, so a legitimate fraction in the tab must not be reported as wrong.
    """
    m = NAME_RANGE.search(name)
    if m:
        a, a_ms, b, b_ms = m.groups()
        # The name is always printed as seconds.fraction, so infer the fraction
        # width from the digits actually present instead of assuming one.
        return float(f"{a}.{a_ms}"), float(f"{b}.{b_ms}"), 0.01
    m = NAME_RANGE_S.search(name)
    if m:
        return float(m.group(1)), float(m.group(2)), 1.0
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tab_file")
    ap.add_argument("--offset", type=float, default=2.0, help="same offset you will pass to download")
    ap.add_argument("--show", type=int, default=0, help="print at most N rows in detail")
    args = ap.parse_args()

    rows = 0
    short = 0
    inverted = 0
    mismatch = 0

    with open(args.tab_file) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                short += 1
                print(f"[MAL] línea {lineno}: {len(parts)} campos, se esperan >=6")
                continue
            rows += 1
            name = os.path.basename(parts[0].replace(".txt", ""))
            clip_id, s_sec, s_ms, e_sec, e_ms = parts[1], *parts[2:6]
            d = divisor_for(name)
            start = round(int(s_sec) + int(s_ms) / d - args.offset, 3)
            end = round(int(e_sec) + int(e_ms) / d + args.offset, 3)

            problems = []
            if not start < end:
                inverted += 1
                problems.append("start >= end")

            ref = name_range(name)
            if ref:
                ref_start, ref_end, tol = ref
                # name holds the un-offset range
                exp_start = round(ref_start - args.offset, 3)
                exp_end = round(ref_end + args.offset, 3)
                if abs(exp_start - start) > tol or abs(exp_end - end) > tol:
                    mismatch += 1
                    problems.append(
                        f"no coincide con el nombre (esperado ~{exp_start}/~{exp_end} con /{d})"
                    )

            if problems:
                print(f"[MAL] línea {lineno}: {name} -> {start}/{end} ({'; '.join(problems)})")
            elif args.show:
                print(f"[OK ] línea {lineno}: /{d} {name} -> {start}/{end}")

    if args.show:
        print()
    print(
        f"filas={rows} mal_formadas={short} invertidas={inverted} "
        f"sin_coincidir_con_nombre={mismatch}"
    )
    if rows == 0:
        print("⚠️  no se leyó ninguna fila: revisa que el fichero sea tabulado")
        return 1
    return 0 if (short or inverted or mismatch) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
