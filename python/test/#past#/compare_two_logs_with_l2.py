#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two FlowTree 1D-NN logs (different seeds) for the same doc_id,
and compute L2 distance in the original embedding space (vocab.npy).

What it does:
  - Parse each log and extract rows under "=== + -> -  scan NN ==="
    for a specified doc_id block:
      u_id, u_x, v_id, v_x, d1
  - For each u_id (common set; typically 34), compute:
      L2(u_id, v_id_seed110), L2(u_id, v_id_seed111)
    using vocab.npy vectors (float32).
  - Print a table keyed by u_id with both seeds side-by-side.

Usage:
  python3 compare_two_logs_with_l2.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --logA "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed110.log" --seedA 110 \
    --logB "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed111.log" --seedB 111 \
    --doc_id 0

Notes:
  - Assumes logs contain a block starting with:
      "=== doc_id=0 idx=... dump_seed=... ==="
    and within that block a section:
      "=== + -> -  scan NN ==="
    followed by rows:
      u_id  u_x  v_id  v_x  d1
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np


# ----------------------------
# Parsing
# ----------------------------

DOC_HEADER_RE = re.compile(r"^===\s*doc_id=(\d+)\s+idx=(\d+)\s+dump_seed=([-\d]+)\s+packedN=(\d+)\s+H=(\d+)\s*===\s*$")
NN_SECTION_RE = re.compile(r"^===\s*\+\s*->\s*-\s*scan\s*NN\s*===\s*$")
# Row format in your logs: "u_id  u_x   ->   v_id  v_x   d1"
NN_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


@dataclass
class NNRow:
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int


def extract_doc_block(text: str, doc_id: int) -> Optional[str]:
    """Return the substring of the log corresponding to the first block for doc_id."""
    lines = text.splitlines()
    starts: List[int] = []
    headers: List[Tuple[int, int]] = []  # (line_index, doc_id)
    for i, line in enumerate(lines):
        m = DOC_HEADER_RE.match(line.strip())
        if m:
            headers.append((i, int(m.group(1))))
    if not headers:
        return None

    # Find first header matching doc_id
    target_idx = None
    for k, (i, did) in enumerate(headers):
        if did == doc_id:
            target_idx = k
            break
    if target_idx is None:
        return None

    start_line = headers[target_idx][0]
    end_line = headers[target_idx + 1][0] if (target_idx + 1) < len(headers) else len(lines)
    return "\n".join(lines[start_line:end_line])


def parse_plus_to_minus_nn(block_text: str) -> Dict[int, NNRow]:
    """
    Parse a doc block and return dict u_id -> NNRow for the '+ -> - scan NN' section.
    """
    lines = block_text.splitlines()
    # Find NN section start
    sec_start = None
    for i, line in enumerate(lines):
        if NN_SECTION_RE.match(line.strip()):
            sec_start = i
            break
    if sec_start is None:
        return {}

    rows: Dict[int, NNRow] = {}
    # Read subsequent lines until blank line or next header-ish line
    for line in lines[sec_start + 1 :]:
        s = line.strip()
        if not s:
            # stop at first blank line after table (your logs put blank line before "[check]")
            break
        if s.startswith("[check") or s.startswith("==="):
            break

        # The printed table sometimes has "->" column in between; your raw log lines are already aligned.
        # In your logs, the rows are effectively: u_id u_x v_id v_x d1 (no '->' token).
        m = NN_ROW_RE.match(s.replace("->", " ").replace("|", " "))
        if not m:
            # Some logs might contain header line "u_id u_x -> v_id v_x d1"; skip it.
            continue
        u_id, u_x, v_id, v_x, d1 = map(int, m.groups())
        rows[u_id] = NNRow(u_id=u_id, u_x=u_x, v_id=v_id, v_x=v_x, d1=d1)
    return rows


# ----------------------------
# L2 computation
# ----------------------------

def l2_distance(vocab: np.ndarray, i: int, j: int) -> float:
    # vocab is (V,D) float32 mmap
    a = vocab[i]
    b = vocab[j]
    # float32 -> float64 accumulation for stability
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.dot(d, d)))


# ----------------------------
# Printing
# ----------------------------

def fmt_float(x: Optional[float], w: int = 10, prec: int = 6) -> str:
    if x is None:
        return " " * w
    s = f"{x:.{prec}f}"
    return s.rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing vocab.npy")
    ap.add_argument("--logA", required=True, help="Path to log file for seedA")
    ap.add_argument("--seedA", required=True, type=int, help="Seed label for logA (for printing)")
    ap.add_argument("--logB", required=True, help="Path to log file for seedB")
    ap.add_argument("--seedB", required=True, type=int, help="Seed label for logB (for printing)")
    ap.add_argument("--doc_id", required=True, type=int, help="doc_id to parse within logs")
    ap.add_argument("--out", default="", help="Optional output text file path")
    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.npy not found: {vocab_path}")

    textA = open(args.logA, "r", encoding="utf-8", errors="ignore").read()
    textB = open(args.logB, "r", encoding="utf-8", errors="ignore").read()

    blockA = extract_doc_block(textA, args.doc_id)
    blockB = extract_doc_block(textB, args.doc_id)
    if blockA is None:
        raise RuntimeError(f"doc_id={args.doc_id} block not found in logA: {args.logA}")
    if blockB is None:
        raise RuntimeError(f"doc_id={args.doc_id} block not found in logB: {args.logB}")

    rowsA = parse_plus_to_minus_nn(blockA)
    rowsB = parse_plus_to_minus_nn(blockB)

    if not rowsA:
        raise RuntimeError(f"No '+ -> - scan NN' rows found in logA for doc_id={args.doc_id}")
    if not rowsB:
        raise RuntimeError(f"No '+ -> - scan NN' rows found in logB for doc_id={args.doc_id}")

    # Common u_ids (you expect 34)
    common_u = sorted(set(rowsA.keys()) & set(rowsB.keys()))
    if not common_u:
        raise RuntimeError("No common u_id between the two logs (unexpected)")

    # Load vocab with mmap (fast)
    vocab = np.load(vocab_path, mmap_mode="r")
    V = vocab.shape[0]

    # Compute L2 for both
    out_lines: List[str] = []
    out_lines.append(f"=== doc_id={args.doc_id} compare seeds {args.seedA} vs {args.seedB} ===")
    out_lines.append(f"common u count = {len(common_u)}")
    out_lines.append("")
    out_lines.append("[table] u-keyed: (seedA) u_x, v_id, v_x, d1, L2   ||   (seedB) u_x, v_id, v_x, d1, L2")
    out_lines.append(
        " u_id |"
        f"  u_x({args.seedA:>3})  v_id({args.seedA:>3})  v_x({args.seedA:>3})  d1({args.seedA:>3})      L2({args.seedA:>3})"
        "  ||"
        f"  u_x({args.seedB:>3})  v_id({args.seedB:>3})  v_x({args.seedB:>3})  d1({args.seedB:>3})      L2({args.seedB:>3})"
    )
    out_lines.append("-" * 120)

    for u in common_u:
        a = rowsA[u]
        b = rowsB[u]

        # sanity
        if u < 0 or u >= V:
            l2a = None
            l2b = None
        else:
            l2a = l2_distance(vocab, u, a.v_id) if (0 <= a.v_id < V) else None
            l2b = l2_distance(vocab, u, b.v_id) if (0 <= b.v_id < V) else None

        out_lines.append(
            f"{u:5d} |"
            f"{a.u_x:8d} {a.v_id:10d} {a.v_x:8d} {a.d1:7d} {fmt_float(l2a, w=12)}"
            "  ||"
            f"{b.u_x:8d} {b.v_id:10d} {b.v_x:8d} {b.d1:7d} {fmt_float(l2b, w=12)}"
        )

    out_lines.append("")
    # Optional quick summary: how many picked different v
    diff = sum(1 for u in common_u if rowsA[u].v_id != rowsB[u].v_id)
    out_lines.append(f"[summary] different v_id count = {diff} / {len(common_u)}")
    out_lines.append("")

    result = "\n".join(out_lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
    print(result)


if __name__ == "__main__":
    main()
