#!/usr/bin/env python3
# nn_compare_1d_0123.py  (batch-capable, OVERALL-summary-only mode added)

import os
import re
import glob
import argparse
import numpy as np


# -------------------------
# parsing: embed1d_algoA_demo.py output (*.out)
# expects lines like:
#   v47  (id= 0)  x=  0  depth=2
# -------------------------
_LEAF_RE = re.compile(
    r'^\s*(?P<name>[A-Za-z0-9_]+)\s+\(id=\s*(?P<id>\d+)\)\s+x=\s*(?P<x>-?\d+)\s+depth=\s*(?P<depth>-?\d+)\s*$'
)

def parse_embed_out(path: str):
    """
    Returns:
      f: dict[vocab_id -> x]
      meta: dict[vocab_id -> (node_id, depth, name)]
    Only keeps leaves with name like v{int} (e.g., v47).
    """
    f = {}
    meta = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            m = _LEAF_RE.match(line.rstrip("\n"))
            if not m:
                continue
            name = m.group("name")
            node_id = int(m.group("id"))
            x = int(m.group("x"))
            depth = int(m.group("depth"))

            if len(name) >= 2 and name[0] == "v" and name[1:].isdigit():
                vid = int(name[1:])
                f[vid] = x
                meta[vid] = (node_id, depth, name)

    if not f:
        raise RuntimeError(f"[parse] no v* leaves found in: {path}")
    return f, meta


def l2_dist(V, u: int, v: int) -> float:
    return float(np.linalg.norm(V[u] - V[v]))


def nn_by_1d(f, V, S, u: int):
    """
    pick NN by (|x_u-x_v|, L2(u,v), v) lexicographically
    """
    cand = []
    xu = f[u]
    for v in S:
        if v == u:
            continue
        cand.append((abs(xu - f[v]), l2_dist(V, u, v), v))
    cand.sort()
    d1, d2, v = cand[0]
    return v, int(d1), float(d2)


def nn_by_l2(V, S, u: int):
    """
    pick NN by (L2(u,v), v)
    """
    cand = []
    for v in S:
        if v == u:
            continue
        cand.append((l2_dist(V, u, v), v))
    cand.sort()
    d2, v = cand[0]
    return v, float(d2)


def summarize_one_pair(V, fA, fB, S):
    """
    Summary over vertices in S:
      - BOTH: A-NN == B-NN
      - A=L2: A-NN == L2-NN
      - B=L2: B-NN == L2-NN
      - BOTH&L2: A-NN == B-NN == L2-NN
    Returns dict with counts and ratios.
    """
    n = len(S)
    both = 0
    a_l2 = 0
    b_l2 = 0
    both_l2 = 0

    for u in S:
        a_v, _, _ = nn_by_1d(fA, V, S, u)
        b_v, _, _ = nn_by_1d(fB, V, S, u)
        l2_v, _ = nn_by_l2(V, S, u)

        if a_v == b_v:
            both += 1
        if a_v == l2_v:
            a_l2 += 1
        if b_v == l2_v:
            b_l2 += 1
        if (a_v == b_v) and (a_v == l2_v):
            both_l2 += 1

    def ratio(x):
        return 100.0 * x / n if n else 0.0

    return {
        "n": n,
        "both": both,
        "diff": n - both,
        "both_pct": ratio(both),
        "a_l2": a_l2,
        "a_l2_pct": ratio(a_l2),
        "b_l2": b_l2,
        "b_l2_pct": ratio(b_l2),
        "both_l2": both_l2,
        "both_l2_pct": ratio(both_l2),
    }


def print_summary_line(tag: str, summ: dict):
    print(
        f"[{tag}] n={summ['n']}  "
        f"BOTH={summ['both']}({summ['both_pct']:.1f}%)  "
        f"A=L2={summ['a_l2']}({summ['a_l2_pct']:.1f}%)  "
        f"B=L2={summ['b_l2']}({summ['b_l2_pct']:.1f}%)  "
        f"BOTH&L2={summ['both_l2']}({summ['both_l2_pct']:.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir containing vocab.npy")
    ap.add_argument("--A_out", default=None, help="single A embed output (.out)")
    ap.add_argument("--B_out", default=None, help="single B embed output (.out)")
    ap.add_argument("--A_dir", default=None, help="batch dir for A outputs")
    ap.add_argument("--B_dir", default=None, help="batch dir for B outputs")
    ap.add_argument("--glob", dest="glob_pat", default="adj_*.out",
                    help="pattern inside A_dir/B_dir (default: adj_*.out)")
    ap.add_argument("--only_overall", action="store_true",
                    help="(batch mode) print ONLY overall summary (no per-file lines)")
    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    if not os.path.exists(vocab_path):
        raise RuntimeError(f"missing vocab.npy: {vocab_path}")
    V = np.load(vocab_path)

    # -------- single mode --------
    if args.A_out and args.B_out:
        fA, _ = parse_embed_out(args.A_out)
        fB, _ = parse_embed_out(args.B_out)

        S = sorted(set(fA.keys()) & set(fB.keys()))
        if not S:
            raise RuntimeError("no common v* leaves between A_out and B_out")

        summ = summarize_one_pair(V, fA, fB, S)
        print_summary_line("SINGLE", summ)
        return

    # -------- batch mode --------
    if not (args.A_dir and args.B_dir):
        raise RuntimeError("use either (--A_out --B_out) or (--A_dir --B_dir)")

    A_files = sorted(glob.glob(os.path.join(args.A_dir, args.glob_pat)))
    B_files = sorted(glob.glob(os.path.join(args.B_dir, args.glob_pat)))

    A_map = {os.path.basename(p): p for p in A_files}
    B_map = {os.path.basename(p): p for p in B_files}
    common = sorted(set(A_map.keys()) & set(B_map.keys()))
    if not common:
        raise RuntimeError("no matching .out basenames between A_dir and B_dir")

    # aggregate arrays (percent per file)
    both_pcts = []
    a_l2_pcts = []
    b_l2_pcts = []
    both_l2_pcts = []
    n_files = 0
    skipped = 0

    for base in common:
        a_path = A_map[base]
        b_path = B_map[base]
        try:
            fA, _ = parse_embed_out(a_path)
            fB, _ = parse_embed_out(b_path)
            S = sorted(set(fA.keys()) & set(fB.keys()))
            if not S:
                skipped += 1
                continue

            summ = summarize_one_pair(V, fA, fB, S)

            if not args.only_overall:
                print_summary_line(base, summ)

            both_pcts.append(summ["both_pct"])
            a_l2_pcts.append(summ["a_l2_pct"])
            b_l2_pcts.append(summ["b_l2_pct"])
            both_l2_pcts.append(summ["both_l2_pct"])
            n_files += 1
        except Exception:
            skipped += 1
            continue

    if n_files == 0:
        raise RuntimeError("no valid pairs processed")

    def stat(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.min()), float(arr.max())

    m_both, lo_both, hi_both = stat(both_pcts)
    m_a, lo_a, hi_a = stat(a_l2_pcts)
    m_b, lo_b, hi_b = stat(b_l2_pcts)
    m_bl2, lo_bl2, hi_bl2 = stat(both_l2_pcts)

    print("=== Batch summary (overall) ===")
    print(f"#pairs_total = {len(common)}  processed = {n_files}  skipped = {skipped}")
    print(f"BOTH%     : mean={m_both:.1f}  min={lo_both:.1f}  max={hi_both:.1f}")
    print(f"A=L2%     : mean={m_a:.1f}  min={lo_a:.1f}  max={hi_a:.1f}")
    print(f"B=L2%     : mean={m_b:.1f}  min={lo_b:.1f}  max={hi_b:.1f}")
    print(f"BOTH&L2%  : mean={m_bl2:.1f}  min={lo_bl2:.1f}  max={hi_bl2:.1f}")


if __name__ == "__main__":
    main()
