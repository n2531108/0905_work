#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_flowtree_time.py

目的:
  - ot_estimators.OTEstimators を1回だけ初期化して（load_vocabulary + tree build）
  - flowtree_distance_pair を「多数 doc」に対して繰り返し呼び、時間を計測する
  - delta(差分質量) / raw(元の正規化測度) / both を選択可能
  - 平均・中央値・p95・min/max と、1callあたりの時間も表示する

例:
  # doc 0..99 を delta で (100 docs × 50周 = 5000 calls) 計測
  python3 bench_flowtree_time.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --so_dir   "/mnt/c/Users/成見/0905_work/native/build" \
    --qid 0 --docs 0:99 --seed 110 \
    --mode delta \
    --warmup 5 --repeat_outer 50

  # raw と delta を両方
  python3 bench_flowtree_time.py ... --mode both
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple


import numpy as np


# ----------------------------
# measure helpers
# ----------------------------

def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    ids = [int(x) for x in ids]
    if not ids:
        return []
    w = 1.0 / float(len(ids))
    return [(vid, w) for vid in ids]


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
    out = [(vid, w / s) for vid, w in acc.items()]
    out.sort(key=lambda x: x[0])
    return out


def to_measure(obj: Any) -> List[Tuple[int, float]]:
    """
    queries.npy / dataset.npy の要素を list[(vid, w)] に変換（正規化込み）
      - list[int] / ndarray(1D): 一様重み
      - list[(id,w)] / ndarray(N,2): 正規化
      - object ndarray: tolist() してから処理
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        elif obj.ndim == 1:
            try:
                ids = [int(x) for x in obj.tolist()]
                return _uniform_measure(ids)
            except Exception:
                pass
        elif obj.ndim == 2 and obj.shape[1] == 2:
            out = [(int(a), float(b)) for a, b in obj.tolist()]
            return _normalize_measure(out)

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], (int, np.integer)):
            return _uniform_measure([int(x) for x in obj])

        out2: List[Tuple[int, float]] = []
        ok = True
        for x in obj:
            if isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 2:
                out2.append((int(x[0]), float(x[1])))
            else:
                ok = False
                break
        if ok:
            return _normalize_measure(out2)

    raise TypeError(f"Unsupported measure format: type={type(obj)}")


def measure_to_dict(m: List[Tuple[int, float]]) -> Dict[int, float]:
    acc: Dict[int, float] = {}
    for vid, w in m:
        vid = int(vid)
        acc[vid] = acc.get(vid, 0.0) + float(w)
    return acc


def dict_to_measure(d: Dict[int, float], eps: float = 1e-15) -> List[Tuple[int, float]]:
    out = [(int(k), float(v)) for k, v in d.items() if float(v) > eps]
    out.sort(key=lambda x: x[0])
    return out


def sum_measure(m: List[Tuple[int, float]]) -> float:
    return float(sum(w for _, w in m))


def build_delta_masses(
    q_measure: List[Tuple[int, float]],
    d_measure: List[Tuple[int, float]],
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    delta = q - d
    plus  = max(delta, 0)
    minus = max(-delta, 0)
    delta_total = sum(plus) == sum(minus) (理想)
    """
    qd = measure_to_dict(q_measure)
    dd = measure_to_dict(d_measure)
    vids = set(qd.keys()) | set(dd.keys())

    plus: Dict[int, float] = {}
    minus: Dict[int, float] = {}
    eps = 1e-15
    for vid in vids:
        delta = float(qd.get(vid, 0.0) - dd.get(vid, 0.0))
        if delta > eps:
            plus[int(vid)] = delta
        elif delta < -eps:
            minus[int(vid)] = -delta

    delta_total = float(sum(plus.values()))
    return plus, minus, delta_total


def balance_plus_minus(
    plus: List[Tuple[int, float]],
    minus: List[Tuple[int, float]],
    balance_eps: float = 1e-9,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    数値誤差で sum(plus) != sum(minus) になったら小さい方に合わせてスケールする。
    """
    s1 = sum_measure(plus)
    s2 = sum_measure(minus)
    if s1 <= 0 or s2 <= 0:
        return plus, minus
    if abs(s1 - s2) <= balance_eps:
        return plus, minus
    if s1 > s2:
        scale = s2 / s1
        plus = [(vid, w * scale) for vid, w in plus]
    else:
        scale = s1 / s2
        minus = [(vid, w * scale) for vid, w in minus]
    return plus, minus


# ----------------------------
# CLI parsing for docs
# ----------------------------

def parse_docs_spec(spec: str) -> List[int]:
    """
    spec:
      - "0:99" or "0:99:2" (end inclusive)
      - "5" single
      - "1,2,10"
    """
    spec = spec.strip()
    if "," in spec:
        out = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out

    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid --docs spec: {spec}")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("step must be non-zero")
        # inclusive end
        if step > 0:
            return list(range(start, end + 1, step))
        else:
            return list(range(start, end - 1, step))

    return [int(spec)]


# ----------------------------
# timing stats
# ----------------------------

def pct(xs: List[float], p: float) -> float:
    xs2 = sorted(xs)
    if not xs2:
        return float("nan")
    if p <= 0:
        return xs2[0]
    if p >= 100:
        return xs2[-1]
    k = (len(xs2) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs2[lo]
    w = k - lo
    return xs2[lo] * (1 - w) + xs2[hi] * w


def print_stats(name: str, xs: List[float]) -> None:
    xs = [float(x) for x in xs if (x is not None and not math.isnan(x) and not math.isinf(x))]
    if not xs:
        print(f"[{name}] no data")
        return
    print(f"[{name}] n={len(xs)}")
    print(f"  mean  = {statistics.fmean(xs):.6f} s")
    print(f"  median= {statistics.median(xs):.6f} s")
    print(f"  p95   = {pct(xs, 95):.6f} s")
    print(f"  min   = {min(xs):.6f} s")
    print(f"  max   = {max(xs):.6f} s")


# ----------------------------
# main
# ----------------------------

def import_ot_estimators(so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    import ot_estimators  # type: ignore
    return ot_estimators


def prepare_pair_mode_if_possible(ot) -> None:
    """
    あなたの C++ が prepare_pair_mode を持っていれば使う。
    無ければ stage=2 にするための dummy load_dataset を入れる。
    """
    if hasattr(ot, "prepare_pair_mode"):
        ot.prepare_pair_mode()
        return
    dummy = [[(0, 1.0)]]
    ot.load_dataset(dummy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--so_dir", required=True)
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--docs", required=True, help='doc spec: "0:99" (inclusive) / "0:99:2" / "1,2,3" / "5"')
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["delta", "raw", "both"], default="delta")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat_outer", type=int, default=50)
    ap.add_argument("--balance_eps", type=float, default=1e-9)
    args = ap.parse_args()

    doc_ids = parse_docs_spec(args.docs)
    if not doc_ids:
        raise ValueError("empty doc list")

    # ----------------------------
    # load npys
    # ----------------------------
    t0 = time.perf_counter()
    queries = np.load(os.path.join(args.data_dir, "queries.npy"), allow_pickle=True)
    dataset = np.load(os.path.join(args.data_dir, "dataset.npy"), allow_pickle=True)
    vocab = np.load(os.path.join(args.data_dir, "vocab.npy"), mmap_mode="r")
    t_load_npys = time.perf_counter() - t0

    q_measure = to_measure(queries[args.qid])

    # pre-build doc measures
    doc_measures_raw: Dict[int, List[Tuple[int, float]]] = {}
    for did in doc_ids:
        doc_measures_raw[int(did)] = to_measure(dataset[int(did)])

    # show one example
    d0 = doc_ids[0]
    print("[data]")
    print(f"  qid={args.qid} docs={args.docs} (count={len(doc_ids)}) seed={args.seed}")
    print(f"  vocab.shape={tuple(vocab.shape)} dtype={vocab.dtype}")
    print(f"  |q|={len(q_measure)} sum(q)={sum_measure(q_measure):.6f} (normalized)")
    print(f"  example doc_id={d0}: |d|={len(doc_measures_raw[d0])} sum(d)={sum_measure(doc_measures_raw[d0]):.6f} (normalized)")

    # precompute delta per doc if needed
    doc_plus: Dict[int, List[Tuple[int, float]]] = {}
    doc_minus: Dict[int, List[Tuple[int, float]]] = {}
    doc_delta_total: Dict[int, float] = {}
    if args.mode in ("delta", "both"):
        for did in doc_ids:
            plus_d, minus_d, delta_total = build_delta_masses(q_measure, doc_measures_raw[did])
            plus_m = dict_to_measure(plus_d)
            minus_m = dict_to_measure(minus_d)
            plus_m, minus_m = balance_plus_minus(plus_m, minus_m, balance_eps=args.balance_eps)
            doc_plus[did] = plus_m
            doc_minus[did] = minus_m
            doc_delta_total[did] = float(sum_measure(plus_m))

        # summary of delta for the first doc
        print(f"  example delta_total(doc={d0}) = {doc_delta_total[d0]:.6f} (== sum(minus))")

    # ----------------------------
    # import so + init ot
    # ----------------------------
    t0 = time.perf_counter()
    ot_estimators = import_ot_estimators(args.so_dir)
    t_import_so = time.perf_counter() - t0

    ot = ot_estimators.OTEstimators()

    t0 = time.perf_counter()
    ot.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(args.seed))
    prepare_pair_mode_if_possible(ot)
    t_load_vocab_tree = time.perf_counter() - t0

    if not hasattr(ot, "flowtree_distance_pair"):
        raise RuntimeError("OTEstimators has no flowtree_distance_pair")

    # ----------------------------
    # warmup
    # ----------------------------
    def run_one_call_raw(did: int) -> float:
        a = q_measure
        b = doc_measures_raw[did]
        return float(ot.flowtree_distance_pair(a, b))

    def run_one_call_delta(did: int) -> float:
        a = doc_plus[did]
        b = doc_minus[did]
        if sum_measure(a) <= 0.0 or sum_measure(b) <= 0.0:
            return 0.0
        return float(ot.flowtree_distance_pair(a, b))

    # warmup (just cycle docs)
    for _ in range(max(0, args.warmup)):
        for did in doc_ids:
            if args.mode == "raw":
                _ = run_one_call_raw(did)
            elif args.mode == "delta":
                _ = run_one_call_delta(did)
            else:
                _ = run_one_call_delta(did)
                _ = run_one_call_raw(did)

    # ----------------------------
    # benchmark loops
    # ----------------------------
    times_delta: List[float] = []
    times_raw: List[float] = []

    t0_all = time.perf_counter()

    for _ in range(max(1, args.repeat_outer)):
        for did in doc_ids:
            if args.mode in ("delta", "both"):
                t0 = time.perf_counter()
                _ = run_one_call_delta(did)
                times_delta.append(time.perf_counter() - t0)

            if args.mode in ("raw", "both"):
                t0 = time.perf_counter()
                _ = run_one_call_raw(did)
                times_raw.append(time.perf_counter() - t0)

    t_total_calls = time.perf_counter() - t0_all

    # ----------------------------
    # print results
    # ----------------------------
    print()
    print_stats("t_load_npys", [t_load_npys])
    print_stats("t_import_so", [t_import_so])
    print_stats("t_load_vocabulary_build_tree", [t_load_vocab_tree])

    print()
    if args.mode in ("delta", "both"):
        print_stats("t_flowtree_distance_pair_delta", times_delta)
        if times_delta:
            per_call_us = (statistics.fmean(times_delta) * 1e6)
            print(f"  -> mean per-call (delta) = {per_call_us:.3f} µs")

    if args.mode in ("raw", "both"):
        print_stats("t_flowtree_distance_pair_raw", times_raw)
        if times_raw:
            per_call_us = (statistics.fmean(times_raw) * 1e6)
            print(f"  -> mean per-call (raw)   = {per_call_us:.3f} µs")

    print()
    print("[total]")
    print(f"  calls_total = {len(times_delta) + len(times_raw)}")
    print(f"  wall_time_for_calls = {t_total_calls:.6f} s")
    if (len(times_delta) + len(times_raw)) > 0:
        print(f"  wall_time_per_call  = {(t_total_calls / (len(times_delta) + len(times_raw))) * 1e6:.3f} µs")


if __name__ == "__main__":
    main()