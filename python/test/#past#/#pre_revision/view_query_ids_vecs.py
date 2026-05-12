#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show one query's word ID list and its embedding vectors (GloVe 50d assumed).
- Loads: vocab.npy (V,50), vocab_words.npy (V,), queries.npy (Q,)
- Prints: IDs / words / first N dims of vectors to console
- Saves : (optional) TSV with all tokens and all 50 dims

Default behavior changed:
- Console shows 50 IDs/words by default (use --head to change)
- Console uses fixed-point (no scientific notation), 6 decimals

Usage examples:
  python3 python/test/view_query_ids_vecs.py \
    --data_dir /path/to/otdata_glove50_full \
    --qi 0

  # TSV保存（ファイル名だけ渡すと python/test に保存）
  python3 python/test/view_query_ids_vecs.py \
    --data_dir /path/to/otdata_glove50_full \
    --qi 0 --save_tsv query_0000_ids_words_vecs.tsv
"""
import os
import csv
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Show one query's word IDs and 50d vectors (from vocab.npy / vocab_words.npy)."
    )
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing vocab.npy, vocab_words.npy, queries.npy")
    ap.add_argument("--qi", type=int, default=0,
                    help="Query index to inspect (0-based)")
    ap.add_argument("--head", type=int, default=5,
                    help="How many tokens to show inline in console (default 50)")
    ap.add_argument("--vec_show", type=int, default=5,
                    help="How many dims to show per vector in console (default 50)")
    ap.add_argument("--save_tsv", default="",
                    help="If set, save all tokens+all dims to TSV (filename only -> saved under python/test)")
    args = ap.parse_args()

    # ---- make console prints fixed-point, no scientific notation (6 decimals) ----
    np.set_printoptions(suppress=True,
                        formatter={'float_kind': lambda x: f"{x:.6f}"})

    # ---- load files ----
    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    words_path = os.path.join(args.data_dir, "vocab_words.npy")
    queries_path = os.path.join(args.data_dir, "queries.npy")

    if not os.path.isfile(vocab_path):
        raise SystemExit(f"not found: {vocab_path}")
    if not os.path.isfile(words_path):
        raise SystemExit(f"not found: {words_path}")
    if not os.path.isfile(queries_path):
        raise SystemExit(f"not found: {queries_path}")

    vocab = np.load(vocab_path)  # (V, 50)
    words = np.load(words_path, allow_pickle=True)  # (V,)
    queries = np.load(queries_path, allow_pickle=True)  # (Q,)

    if args.qi < 0 or args.qi >= len(queries):
        raise SystemExit(f"qi out of range (0..{len(queries)-1})")

    # ---- fetch one query ----
    ids = np.array(queries[args.qi], dtype=int)  # ID list
    vecs = vocab[ids]                             # (len(ids), 50)

    # ---- console summary ----
    show_n = min(args.head, len(ids))
    print(f"[q={args.qi}] len={len(ids)}")
    print("ID列（先頭N）:", ids[:show_n], "..." if len(ids) > show_n else "")
    print("語（先頭N）   :", [str(words[i]) for i in ids[:show_n]], "..." if len(ids) > show_n else "")
    print("ベクトル形状  :", vecs.shape)

    # clamp vec_show to valid range
    d = vecs.shape[1]
    vec_show = max(1, min(args.vec_show, d))

    print(f"先頭3語のベクトル（各{d}次元のうち先頭{vec_show}要素を表示）")
    for k in range(min(3, len(ids))):
        print(f"  id={ids[k]:6d}  word={str(words[ids[k]]):>12s}  vec[:{vec_show}]={vecs[k,:vec_show]}")

    # ---- optional TSV dump (always saves ALL dims) ----
    if args.save_tsv:
        out_path = args.save_tsv
        # If only filename is given, save under the script directory (python/test)
        if os.path.basename(out_path) == out_path:
            out_path = os.path.join(os.path.dirname(__file__), out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["rank", "id", "word"] + [f"dim{j}" for j in range(d)])
            for r, (i, v) in enumerate(zip(ids, vecs), start=1):
                w.writerow([r, int(i), str(words[i])] + [float(x) for x in v])

        print(f"[saved] {out_path}  rows={len(ids)}  dims={d}")


if __name__ == "__main__":
    main()
