#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np

# -------------------------
# Parsing: embed1d_algoA_demo.py output
# -------------------------
LEAF_LINE_RE = re.compile(
    r"""^\s*
        (?P<name>\S+)
        \s+\(id=\s*(?P<id>-?\d+)\)
        \s+x=\s*(?P<x>-?\d+)
        \s+depth=\s*(?P<depth>-?\d+)
        \s*$
    """,
    re.VERBOSE,
)

def parse_embed_out(path):
    """
    Returns:
      f: dict[vocab_id] = x
      meta: dict (for debug)
    We only keep leaves whose printed name looks like v<integer> (e.g., v47, v0).
    """
    f = {}
    lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()

    in_leaf_block = False
    parsed = 0
    for ln in lines:
        # start marker line is like: "葉(左→右): name, id, x, depth"
        if "葉(左→右)" in ln:
            in_leaf_block = True
            continue
        if not in_leaf_block:
            continue

        m = LEAF_LINE_RE.match(ln)
        if not m:
            # Once leaf block started, allow some non-matching lines but keep going.
            continue

        name = m.group("name").strip()
        x = int(m.group("x"))

        # only accept v<digits>
        if len(name) >= 2 and name[0] == "v" and name[1:].isdigit():
            vid = int(name[1:])
            f[vid] = x
            parsed += 1

    meta = {
        "path": path,
        "parsed_count": parsed,
        "keys": sorted(f.keys()),
    }
    return f, meta

# -------------------------
# Distance utilities
# -------------------------
def l2_dist(V, u, v):
    return float(np.linalg.norm(V[u] - V[v]))

def nearest_by_1d_then_l2(S, f, V, u):
    """
    pick argmin over v!=u of (|f[u]-f[v]|, L2(u,v), v)
    """
    best = None
    for v in S:
        if v == u:
            continue
        tup = (abs(f[u] - f[v]), l2_dist(V, u, v), v)
        if best is None or tup < best:
            best = tup
    d1, dl2, v = best
    return v, d1, dl2

def nearest_by_l2(S, V, u):
    best = None
    for v in S:
        if v == u:
            continue
        tup = (l2_dist(V, u, v), v)
        if best is None or tup < best:
            best = tup
    dl2, v = best
    return v, dl2

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir containing vocab.npy")
    ap.add_argument("--A_out", required=True, help="embedding output for tree A (adj_xxx.out)")
    ap.add_argument("--B_out", required=True, help="embedding output for tree B (adj_xxx.out)")
    ap.add_argument("--use", default="intersect", choices=["intersect", "union"],
                    help="S set from A/B leaves: intersect (default) or union")
    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    V = np.load(vocab_path)

    fA, metaA = parse_embed_out(args.A_out)
    fB, metaB = parse_embed_out(args.B_out)

    SA = set(fA.keys())
    SB = set(fB.keys())
    if args.use == "intersect":
        S = sorted(SA & SB)
    else:
        S = sorted(SA | SB)

    if len(S) == 0:
        raise RuntimeError("No common leaves parsed. Check A_out/B_out formats.")

    # warn if mismatch
    onlyA = sorted(SA - SB)
    onlyB = sorted(SB - SA)
    if onlyA or onlyB:
        print("=== WARNING: leaf set mismatch between A and B ===")
        print(" A-only:", onlyA)
        print(" B-only:", onlyB)
        print(f" Using S = {args.use}: |S|={len(S)}")
        print()

    # summary counters
    both_cnt = 0
    diff_cnt = 0
    l2_match_A_cnt = 0
    l2_match_B_cnt = 0
    l2_match_both_cnt = 0  # (A-NN==L2-NN and B-NN==L2-NN)

    print("=== Per-vertex neighbor list (sorted by dA; marks show A/B/L2 nearest) ===")
    print(f"[A_out parsed] {metaA['parsed_count']} leaves  keys={metaA['keys']}")
    print(f"[B_out parsed] {metaB['parsed_count']} leaves  keys={metaB['keys']}")
    print(f"[S] size={len(S)}  S={S}\n")

    for u in S:
        # If union mode and missing in one side, skip safely
        if u not in fA or u not in fB:
            print(f"[u={u}] SKIP (missing in {'A' if u not in fA else ''}{'B' if u not in fB else ''})")
            continue

        a_v, a_d, a_l2 = nearest_by_1d_then_l2(S, fA, V, u)
        b_v, b_d, b_l2 = nearest_by_1d_then_l2(S, fB, V, u)
        l2_v, l2_d = nearest_by_l2(S, V, u)

        status = "BOTH" if a_v == b_v else "DIFF"
        if status == "BOTH":
            both_cnt += 1
        else:
            diff_cnt += 1

        a_is_l2 = (a_v == l2_v)
        b_is_l2 = (b_v == l2_v)
        if a_is_l2:
            l2_match_A_cnt += 1
        if b_is_l2:
            l2_match_B_cnt += 1
        if a_is_l2 and b_is_l2:
            l2_match_both_cnt += 1

        print(f"\n[u={u:>2d}] status={status}  "
              f"A-NN={a_v:>2d}(dA={a_d},L2={a_l2:.3f})  "
              f"B-NN={b_v:>2d}(dB={b_d},L2={b_l2:.3f})  "
              f"L2-NN={l2_v:>2d}(L2={l2_d:.3f})")

        # full list sorted by (dA, L2, dB, v)
        lst = []
        for v in S:
            if v == u:
                continue
            da = abs(fA[u] - fA[v])
            db = abs(fB[u] - fB[v])
            dl2 = l2_dist(V, u, v)
            lst.append((da, dl2, db, v))
        lst.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

        print(" rank mark  v   dA  dB    L2")
        for i, (da, dl2, db, v) in enumerate(lst, start=1):
            markA = (v == a_v)
            markB = (v == b_v)
            markL = (v == l2_v)

            # mark priority: show combination
            if markA and markB and markL:
                mark = "ABL*"
            elif markA and markB:
                mark = "AB* "
            elif markA and markL:
                mark = "AL* "
            elif markB and markL:
                mark = "BL* "
            elif markA:
                mark = "A*  "
            elif markB:
                mark = "B*  "
            elif markL:
                mark = "L*  "
            else:
                mark = "    "

            print(f" {i:>3d} {mark:>4s} {v:>3d} {da:>3d} {db:>3d} {dl2:>6.3f}")

    # summary
    n = len(S)
    both_ratio = both_cnt / n * 100.0
    diff_ratio = diff_cnt / n * 100.0

    print("\n=== Summary over all vertices (in S) ===")
    print(f"#vertices = {n}")
    print(f"A/B BOTH: {both_cnt} ({both_ratio:.1f}%)  |  DIFF: {diff_cnt} ({diff_ratio:.1f}%)")
    print(f"A-NN == L2-NN: {l2_match_A_cnt}/{n} ({(l2_match_A_cnt/n*100.0):.1f}%)")
    print(f"B-NN == L2-NN: {l2_match_B_cnt}/{n} ({(l2_match_B_cnt/n*100.0):.1f}%)")
    print(f"(A-NN == L2-NN) AND (B-NN == L2-NN): {l2_match_both_cnt}/{n} ({(l2_match_both_cnt/n*100.0):.1f}%)")

if __name__ == "__main__":
    main()
