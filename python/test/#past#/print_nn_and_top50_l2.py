#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(A) flowtree_real_1d_nn_pipeline.py の出力にある
    "=== + -> -  scan NN ===" の表（u_id u_x v_id v_x d1）に出たペアだけ
    元空間 L2 距離を計算して一覧表示

(B) それとは無関係に、query[qid] × dataset[docid] の全ペアから
    L2 が近い上位 topk を表示

想定データ:
  data_dir/{queries.npy, dataset.npy, vocab.npy, vocab_words.npy}
  queries.npy, dataset.npy は list[int] の object 配列（重みなし）でもOK
"""

import argparse
import os
import re
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np


ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


def load_ids_object_array(path: str, index: int) -> List[int]:
    arr = np.load(path, allow_pickle=True)
    x = arr[index]
    # 想定: list[int] または ndarray[int]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    return [int(v) for v in list(x)]


def parse_nn_pairs_from_text(text: str) -> List[Dict[str, int]]:
    """
    "=== + -> -  scan NN ===" 以降の表行をすべて拾う。
    行形式: u_id u_x v_id v_x d1
    """
    pairs = []
    in_table = False
    for line in text.splitlines():
        if "=== + -> -  scan NN ===" in line:
            in_table = True
            continue
        if in_table:
            # 次のセクションに入ったら終了（doc区切りや [check] 行など）
            if line.startswith("[check") or line.startswith("===") or line.strip() == "":
                # 空行は表中にもあり得るので break しない：ただし次の "===" で止める
                if line.startswith("===") or line.startswith("[check"):
                    break
                continue

            m = ROW_RE.match(line)
            if m:
                u_id, u_x, v_id, v_x, d1 = map(int, m.groups())
                pairs.append({"u_id": u_id, "u_x": u_x, "v_id": v_id, "v_x": v_x, "d1": d1})
    return pairs


def l2(vocab: np.ndarray, a: int, b: int) -> float:
    # vocab is memmap float32 (V, D)
    diff = vocab[a] - vocab[b]
    return float(np.sqrt(np.dot(diff, diff)))


def safe_word(words: np.ndarray, vid: int) -> str:
    try:
        return str(words[vid])
    except Exception:
        return "<?>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--qid", type=int, default=0)
    ap.add_argument("--docid", type=int, default=0)
    ap.add_argument("--nn_log", default=None, help="flowtree_real_1d_nn_pipeline.py の標準出力ログ（無ければstdinから読む）")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    data_dir = args.data_dir
    qid = args.qid
    docid = args.docid
    topk = args.topk

    vocab_path = os.path.join(data_dir, "vocab.npy")
    words_path = os.path.join(data_dir, "vocab_words.npy")
    queries_path = os.path.join(data_dir, "queries.npy")
    dataset_path = os.path.join(data_dir, "dataset.npy")

    vocab = np.load(vocab_path, mmap_mode="r")
    words = np.load(words_path, allow_pickle=True)
    q_ids = load_ids_object_array(queries_path, qid)
    d_ids = load_ids_object_array(dataset_path, docid)

    # ---- read text ----
    if args.nn_log:
        text = open(args.nn_log, "r", encoding="utf-8", errors="ignore").read()
    else:
        text = sys.stdin.read()

    nn_pairs = parse_nn_pairs_from_text(text)

    # ---- (A) pairs in NN table only ----
    if not nn_pairs:
        print("[WARN] NN表がログから見つかりませんでした。")
        print("       ログに '=== + -> -  scan NN ===' が含まれているか確認して下さい。")
    else:
        rows = []
        for p in nn_pairs:
            u, v = p["u_id"], p["v_id"]
            dist = l2(vocab, u, v)
            rows.append((dist, p))
        rows.sort(key=lambda x: x[0])

        print("\n" + "="*80)
        print(f"(A) NN表に出たペアだけ：元空間L2距離（昇順）  count={len(rows)}")
        print("# rank | l2 | u_id u_word | v_id v_word | d1(u_x-v_x) | (u_x, v_x)")
        for r, (dist, p) in enumerate(rows, 1):
            u, v = p["u_id"], p["v_id"]
            print(f"{r:3d} | {dist:9.6f} | "
                  f"{u:7d} {safe_word(words,u):>15s} | "
                  f"{v:7d} {safe_word(words,v):>15s} | "
                  f"{p['d1']:4d} | ({p['u_x']:3d},{p['v_x']:3d})")

    # ---- (B) top-k among all q_ids x d_ids ----
    Q = vocab[q_ids]  # (Q, D)
    D = vocab[d_ids]  # (D, D)
    dist_mat = np.linalg.norm(Q[:, None, :] - D[None, :, :], axis=2)  # (Q, D)

    # topk extraction without sorting all (still small enough, but do efficient-ish)
    # flatten indices
    flat = dist_mat.ravel()
    k = min(topk, flat.size)
    idx = np.argpartition(flat, k-1)[:k]
    idx = idx[np.argsort(flat[idx])]

    # quick lookup: whether pair is in NN table
    nn_set = set()
    for p in nn_pairs:
        nn_set.add((p["u_id"], p["v_id"]))

    print("\n" + "="*80)
    print(f"(B) query[{qid}]×doc[{docid}] 全ペアから L2 が近い上位{topk}（重み無関係）")
    print(f"# query size={len(q_ids)} doc size={len(d_ids)} pairs={flat.size}")
    print("# rank | l2 | q_vid q_word | d_vid d_word | in_NN_table")
    Qn = len(q_ids)
    Dn = len(d_ids)
    for r, flat_i in enumerate(idx, 1):
        i = int(flat_i // Dn)
        j = int(flat_i % Dn)
        qv = int(q_ids[i])
        dv = int(d_ids[j])
        mark = "YES" if (qv, dv) in nn_set else "no"
        print(f"{r:3d} | {float(flat[flat_i]):9.6f} | "
              f"{qv:7d} {safe_word(words,qv):>15s} | "
              f"{dv:7d} {safe_word(words,dv):>15s} | {mark}")

    print("\n[done]")

if __name__ == "__main__":
    main()
