#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_flowtree_baseline_old_vs_new.py

目的:
  同じ query / doc / seed に対して、

    1. 元の ot_estimators の flowtree_rank(...) で得る score
    2. 新しい ot_estimators_twotree の flowtree_distance_pair(...) で得る score

  を比較する。

これにより、
  - 新しい C++ 側の pair 経路が元の ranking 経路と一致しているか
を切り分ける。

使い方:
  python3 compare_flowtree_baseline_old_vs_new.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --so_dir_old "/mnt/c/Users/成見/0905_work/native/build" \
    --so_dir_new "/mnt/c/Users/成見/0905_work/native/build" \
    --qid 0 \
    --doc_id 0 \
    --seed 110
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# 1) measure loader
# ============================================================

def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    if len(ids) == 0:
        return []
    w = 1.0 / float(len(ids))
    return [(int(v), w) for v in ids]


def _normalize_measure(m: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not m:
        return []
    acc: Dict[int, float] = {}
    for vid, w in m:
        vid = int(vid)
        w = float(w)
        if w <= 0:
            continue
        acc[vid] = acc.get(vid, 0.0) + w
    if not acc:
        return []
    s = float(sum(acc.values()))
    if s <= 0:
        return []
    return sorted([(vid, w / s) for vid, w in acc.items()], key=lambda x: x[0])


def to_measure(obj: Any) -> List[Tuple[int, float]]:
    if obj is None:
        return []

    if isinstance(obj, (np.integer, int)):
        return [(int(obj), 1.0)]

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return to_measure(obj.tolist())
        if obj.ndim == 2 and obj.shape[1] == 2:
            out = [(int(a), float(b)) for a, b in obj.tolist()]
            return _normalize_measure(out)
        if obj.ndim == 1:
            ids = [int(x) for x in obj.tolist()]
            return _uniform_measure(ids)
        return to_measure(obj.tolist())

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []

        first = obj[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            out = []
            ok = True
            for x in obj:
                if not (isinstance(x, (list, tuple)) and len(x) == 2):
                    ok = False
                    break
                out.append((int(x[0]), float(x[1])))
            if ok:
                return _normalize_measure(out)

        ids = []
        ok = True
        for x in obj:
            if isinstance(x, (np.integer, int)):
                ids.append(int(x))
            else:
                ok = False
                break
        if ok:
            return _uniform_measure(ids)

    raise TypeError(f"Unsupported measure format: {type(obj)}")


def load_all_inputs(data_dir: str):
    vocab = np.load(os.path.join(data_dir, "vocab.npy"), allow_pickle=False)
    dataset = np.load(os.path.join(data_dir, "dataset.npy"), allow_pickle=True)
    queries = np.load(os.path.join(data_dir, "queries.npy"), allow_pickle=True)

    if not isinstance(vocab, np.ndarray) or vocab.ndim != 2 or vocab.dtype != np.float32:
        raise RuntimeError("vocab.npy must be float32 2D")

    dataset_measures = [to_measure(x) for x in dataset.tolist()] if dataset.dtype == object else [to_measure(dataset[i]) for i in range(dataset.shape[0])]
    query_measures = [to_measure(x) for x in queries.tolist()] if queries.dtype == object else [to_measure(queries[i]) for i in range(queries.shape[0])]

    return vocab, dataset_measures, query_measures


# ============================================================
# 2) module import
# ============================================================

def import_module_from_dir(module_name: str, so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    mod = __import__(module_name)
    return mod


# ============================================================
# 3) old path: flowtree_rank for one doc
# ============================================================

def old_flowtree_rank_single_score(
    so_dir_old: str,
    vocab: np.ndarray,
    dataset: List[List[Tuple[int, float]]],
    query: List[Tuple[int, float]],
    doc_id: int,
    seed: int,
) -> float:
    mod_old = import_module_from_dir("ot_estimators", so_dir_old)
    ot_old = mod_old.OTEstimators()

    # 旧モジュールが seed 付き load_vocabulary を持つ場合は使う
    try:
        ot_old.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seed))
        seed_mode = "seeded"
    except TypeError:
        ot_old.load_vocabulary(np.asarray(vocab, dtype=np.float32))
        seed_mode = "unseeded"

    ot_old.load_dataset(dataset)

    input_ids = np.array([doc_id], dtype=np.int32)
    output_ids = np.empty((1,), dtype=np.int32)
    output_scores = np.empty((1,), dtype=np.float32)

    try:
        ot_old.flowtree_rank(query, input_ids, output_ids, output_scores, False)
    except TypeError:
        ot_old.flowtree_rank(query, input_ids, output_ids, output_scores)

    score = float(output_scores[0])
    got_doc = int(output_ids[0])

    return score, got_doc, seed_mode


# ============================================================
# 4) new path: flowtree_distance_pair
# ============================================================

def new_flowtree_distance_pair_score(
    so_dir_new: str,
    vocab: np.ndarray,
    query: List[Tuple[int, float]],
    doc: List[Tuple[int, float]],
    seed: int,
) -> float:
    mod_new = import_module_from_dir("ot_estimators_twotree", so_dir_new)
    ot_new = mod_new.OTEstimators()

    try:
        ot_new.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seed))
        seed_mode = "seeded"
    except TypeError:
        ot_new.load_vocabulary(np.asarray(vocab, dtype=np.float32))
        seed_mode = "unseeded"

    score = float(ot_new.flowtree_distance_pair(query, doc))
    return score, seed_mode


# ============================================================
# 5) main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing vocab.npy / dataset.npy / queries.npy")
    ap.add_argument("--so_dir_old", required=True, help="directory containing old ot_estimators*.so")
    ap.add_argument("--so_dir_new", required=True, help="directory containing new ot_estimators_twotree*.so")
    ap.add_argument("--qid", type=int, required=True, help="query index")
    ap.add_argument("--doc_id", type=int, required=True, help="doc index")
    ap.add_argument("--seed", type=int, required=True, help="seed to use for both old and new")
    args = ap.parse_args()

    vocab, dataset, queries = load_all_inputs(args.data_dir)

    if args.qid < 0 or args.qid >= len(queries):
        raise RuntimeError(f"qid out of range: {args.qid} (queries={len(queries)})")
    if args.doc_id < 0 or args.doc_id >= len(dataset):
        raise RuntimeError(f"doc_id out of range: {args.doc_id} (dataset={len(dataset)})")

    query = queries[args.qid]
    doc = dataset[args.doc_id]

    old_score, got_doc, old_seed_mode = old_flowtree_rank_single_score(
        so_dir_old=args.so_dir_old,
        vocab=vocab,
        dataset=dataset,
        query=query,
        doc_id=args.doc_id,
        seed=args.seed,
    )

    new_score, new_seed_mode = new_flowtree_distance_pair_score(
        so_dir_new=args.so_dir_new,
        vocab=vocab,
        query=query,
        doc=doc,
        seed=args.seed,
    )

    abs_diff = abs(old_score - new_score)
    rel_diff = abs_diff / max(1e-12, abs(old_score))

    print("[compare old vs new flowtree baseline]")
    print(f"  qid      = {args.qid}")
    print(f"  doc_id   = {args.doc_id}")
    print(f"  seed     = {args.seed}")
    print()
    print("[old ot_estimators]")
    print(f"  seed_mode            = {old_seed_mode}")
    print(f"  returned_doc_id      = {got_doc}")
    print(f"  flowtree_rank_score  = {old_score:.9f}")
    print()
    print("[new ot_estimators_twotree]")
    print(f"  seed_mode                  = {new_seed_mode}")
    print(f"  flowtree_distance_pair     = {new_score:.9f}")
    print()
    print("[difference]")
    print(f"  abs_diff = {abs_diff:.9f}")
    print(f"  rel_diff = {rel_diff:.9e}")


if __name__ == "__main__":
    main()