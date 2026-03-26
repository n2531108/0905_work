#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_two_logs_pick_greedy_batch_0313.py  (timed)

docをまとめて回し、prematchの効果と、処理時間の内訳もCSVに記録する。

時間計測:
  t_parse    : log block抽出 + NN table parse
  t_compare  : compare_rows生成（L2計算）
  t_prematch : q/d測度→差分質量→greedy prematch
  t_cost_*   : FlowTree cost計算（baseline/residual × seedA/seedB）
  t_total    : doc 1件あたり合計（上記＋雑務）
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class NNRow:
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int


@dataclass
class DocHeader:
    doc_id: int
    idx: int
    seed: int
    packedN: int
    H: int


@dataclass
class CompareRow:
    u_id: int
    A: NNRow
    l2_A: float
    B: NNRow
    l2_B: float
    pick: str
    picked_row: NNRow
    picked_l2: float
    ord_priority: int


@dataclass
class MatchEdge:
    ord_priority: int
    u_id: int
    pick: str
    v_id: int
    d1_pick: int
    l2_pick: float
    flow: float
    rem_u: float
    rem_v: float


# ============================================================
# Measure loaders
# ============================================================

def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    if not ids:
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
    out = [(vid, w / s) for vid, w in acc.items()]
    out.sort(key=lambda x: x[0])
    return out


def to_measure(obj: Any) -> List[Tuple[int, float]]:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        elif obj.ndim == 1:
            return _uniform_measure([int(x) for x in obj.tolist()])
        elif obj.ndim == 2 and obj.shape[1] == 2:
            return _normalize_measure([(int(a), float(b)) for a, b in obj.tolist()])

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

    raise TypeError(f"Unsupported measure format: {type(obj)}")


def load_data(data_dir: str):
    queries = np.load(os.path.join(data_dir, "queries.npy"), allow_pickle=True)
    dataset = np.load(os.path.join(data_dir, "dataset.npy"), allow_pickle=True)
    vocab = np.load(os.path.join(data_dir, "vocab.npy"), mmap_mode="r")
    return queries, dataset, vocab


def sum_measure(m: List[Tuple[int, float]]) -> float:
    return float(sum(w for _, w in m))


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


# ============================================================
# Log parsing (index all doc blocks once)
# ============================================================

_HEADER_RE = re.compile(
    r"^===\s+doc_id=(?P<doc_id>\d+)\s+idx=(?P<idx>\d+)\s+dump_seed=(?P<seed>-?\d+)\s+packedN=(?P<packedN>\d+)\s+H=(?P<H>\d+)\s+===\s*$",
    re.M,
)
_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def index_log_by_doc(text: str) -> Dict[int, Tuple[DocHeader, str]]:
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError("No doc header found in log.")
    out: Dict[int, Tuple[DocHeader, str]] = {}
    for i, m in enumerate(headers):
        did = int(m.group("doc_id"))
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        hdr = DocHeader(
            doc_id=did,
            idx=int(m.group("idx")),
            seed=int(m.group("seed")),
            packedN=int(m.group("packedN")),
            H=int(m.group("H")),
        )
        out[did] = (hdr, block)
    return out


def parse_nn_table(block: str) -> Dict[int, NNRow]:
    rows = _ROW_RE.findall(block)
    if not rows:
        return {}
    out: Dict[int, NNRow] = {}
    for u_id, u_x, v_id, v_x, d1 in rows:
        u = int(u_id)
        out[u] = NNRow(u_id=u, u_x=int(u_x), v_id=int(v_id), v_x=int(v_x), d1=int(d1))
    return out


# ============================================================
# Core math
# ============================================================

def l2_dist(vocab: np.ndarray, u_id: int, v_id: int) -> float:
    a = vocab[int(u_id)]
    b = vocab[int(v_id)]
    return float(np.linalg.norm(a - b))


def build_compare_rows(nnA: Dict[int, NNRow], nnB: Dict[int, NNRow], vocab: np.ndarray) -> Tuple[List[CompareRow], Dict[str, Any]]:
    uA = set(nnA.keys())
    uB = set(nnB.keys())
    common = sorted(uA & uB)

    same_nn = 0
    diff_nn = 0
    picked_A = 0
    picked_B = 0
    l2_gap_sum = 0.0

    rows: List[CompareRow] = []
    for u in common:
        rA = nnA[u]
        rB = nnB[u]
        l2A = l2_dist(vocab, u, rA.v_id)
        l2B = l2_dist(vocab, u, rB.v_id)

        if rA.v_id == rB.v_id:
            same_nn += 1
        else:
            diff_nn += 1
        l2_gap_sum += abs(l2A - l2B)

        if l2A <= l2B:
            pick = "A"
            picked = rA
            picked_l2 = l2A
            picked_A += 1
        else:
            pick = "B"
            picked = rB
            picked_l2 = l2B
            picked_B += 1

        rows.append(CompareRow(
            u_id=u, A=rA, l2_A=l2A, B=rB, l2_B=l2B,
            pick=pick, picked_row=picked, picked_l2=picked_l2, ord_priority=-1
        ))

    rows.sort(key=lambda r: (r.A.d1, r.A.u_x, r.u_id))
    for i, r in enumerate(rows, start=1):
        r.ord_priority = i

    summary = {
        "common_u": len(common),
        "same_nn": same_nn,
        "diff_nn": diff_nn,
        "picked_A": picked_A,
        "picked_B": picked_B,
        "avg_abs_l2_gap": (l2_gap_sum / len(common)) if common else float("nan"),
    }
    return rows, summary


def build_delta_masses(q_measure: List[Tuple[int, float]], d_measure: List[Tuple[int, float]]) -> Tuple[Dict[int, float], Dict[int, float]]:
    qd = measure_to_dict(q_measure)
    dd = measure_to_dict(d_measure)
    vids = set(qd.keys()) | set(dd.keys())

    plus: Dict[int, float] = {}
    minus: Dict[int, float] = {}
    eps = 1e-15
    for vid in vids:
        delta = float(qd.get(vid, 0.0)) - float(dd.get(vid, 0.0))
        if delta > eps:
            plus[int(vid)] = delta
        elif delta < -eps:
            minus[int(vid)] = -delta
    return plus, minus


def greedy_prematch(compare_rows: List[CompareRow], plus_in: Dict[int, float], minus_in: Dict[int, float]) -> Tuple[List[MatchEdge], Dict[int, float], Dict[int, float], Dict[str, Any]]:
    rem_plus = dict(plus_in)
    rem_minus = dict(minus_in)

    matched: List[MatchEdge] = []
    total_flow = 0.0
    skipped_v_exhausted = 0

    for r in compare_rows:
        u = int(r.u_id)
        v = int(r.picked_row.v_id)
        mu = float(rem_plus.get(u, 0.0))
        mv = float(rem_minus.get(v, 0.0))
        if mu <= 0.0:
            continue
        if mv <= 0.0:
            skipped_v_exhausted += 1
            continue
        f = min(mu, mv)
        rem_plus[u] = mu - f
        rem_minus[v] = mv - f
        total_flow += f
        matched.append(MatchEdge(
            ord_priority=r.ord_priority,
            u_id=u, pick=r.pick, v_id=v,
            d1_pick=int(r.picked_row.d1), l2_pick=float(r.picked_l2),
            flow=float(f), rem_u=float(rem_plus[u]), rem_v=float(rem_minus[v]),
        ))

    eps = 1e-15
    sum_rem_plus = sum(x for x in rem_plus.values() if x > eps)
    sum_rem_minus = sum(x for x in rem_minus.values() if x > eps)
    nz_plus = sum(1 for x in rem_plus.values() if x > eps)
    nz_minus = sum(1 for x in rem_minus.values() if x > eps)

    stats = {
        "total_flow_sent": float(total_flow),
        "sum_rem_plus": float(sum_rem_plus),
        "sum_rem_minus": float(sum_rem_minus),
        "nonzero_rem_plus": int(nz_plus),
        "nonzero_rem_minus": int(nz_minus),
        "skipped_v_exhausted": int(skipped_v_exhausted),
        "matched_edges": int(len(matched)),
    }
    return matched, rem_plus, rem_minus, stats


def prematch_cost(match_edges: List[MatchEdge]) -> float:
    return float(sum(e.l2_pick * e.flow for e in match_edges))


# ============================================================
# C++ FlowTree bridge
# ============================================================

def import_ot_estimators(so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    import ot_estimators  # type: ignore
    return ot_estimators


def prepare_pair_mode_if_possible(ot) -> None:
    if hasattr(ot, "prepare_pair_mode"):
        ot.prepare_pair_mode()
        return
    dummy = [[(0, 1.0)]]
    ot.load_dataset(dummy)


def residual_flowtree_cost(so_dir: str, vocab: np.ndarray, seed: int, plus_mass: Dict[int, float], minus_mass: Dict[int, float]) -> float:
    plus = dict_to_measure(plus_mass)
    minus = dict_to_measure(minus_mass)

    s1 = sum_measure(plus)
    s2 = sum_measure(minus)
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0

    if abs(s1 - s2) > 1e-9:
        if s1 > s2:
            scale = s2 / s1
            plus = [(vid, w * scale) for vid, w in plus]
        else:
            scale = s1 / s2
            minus = [(vid, w * scale) for vid, w in minus]

    ot_estimators = import_ot_estimators(so_dir)
    ot = ot_estimators.OTEstimators()
    ot.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seed))
    prepare_pair_mode_if_possible(ot)

    if not hasattr(ot, "flowtree_distance_pair"):
        raise RuntimeError("OTEstimators.flowtree_distance_pair not found")

    return float(ot.flowtree_distance_pair(plus, minus))


# ============================================================
# Batch runner
# ============================================================

def parse_docs_range(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        a, b = spec.split(":")
        lo = int(a)
        hi = int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    out: List[int] = []
    for t in spec.split(","):
        t = t.strip()
        if t:
            out.append(int(t))
    return out


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--so_dir", required=True)
    ap.add_argument("--logA", required=True)
    ap.add_argument("--logB", required=True)
    ap.add_argument("--qid", type=int, default=0)
    ap.add_argument("--docs", default="0:99", help="doc range like 0:99 or comma list")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--skip_costs", action="store_true", help="only compute prematch flow stats (fast)")
    args = ap.parse_args()

    docs = parse_docs_range(args.docs)

    # --- load arrays once ---
    queries, dataset, vocab = load_data(args.data_dir)
    if args.qid < 0 or args.qid >= len(queries):
        raise RuntimeError(f"qid out of range: {args.qid} / {len(queries)}")

    # --- read logs once ---
    idxA = index_log_by_doc(read_text(args.logA))
    idxB = index_log_by_doc(read_text(args.logB))

    # --- query measure once ---
    q_measure = to_measure(queries[args.qid])

    # --- CSV fields ---
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "qid", "doc_id",
        "seedA", "seedB",
        "packedN_A", "packedN_B",
        "common_u", "same_nn", "diff_nn", "picked_A", "picked_B", "avg_abs_l2_gap",
        "delta_total",
        "prematched_flow", "prematch_ratio",
        "residual_mass", "nz_plus", "nz_minus",
        "prematch_cost",
        "baseline_A", "residual_A", "total_A", "improve_A",
        "baseline_B", "residual_B", "total_B", "improve_B",
        # timing
        "t_parse", "t_compare", "t_prematch",
        "t_costA_base", "t_costA_res",
        "t_costB_base", "t_costB_res",
        "t_total",
    ]

    rows_csv: List[Dict[str, Any]] = []

    # timing accumulators
    T_parse: List[float] = []
    T_compare: List[float] = []
    T_prematch: List[float] = []
    T_costA_base: List[float] = []
    T_costA_res: List[float] = []
    T_costB_base: List[float] = []
    T_costB_res: List[float] = []
    T_total: List[float] = []

    for doc_id in docs:
        t0 = time.perf_counter()

        if doc_id not in idxA or doc_id not in idxB:
            continue

        # ---- parse stage ----
        tp0 = time.perf_counter()
        hdrA, blockA = idxA[doc_id]
        hdrB, blockB = idxB[doc_id]
        nnA = parse_nn_table(blockA)
        nnB = parse_nn_table(blockB)
        tp1 = time.perf_counter()
        t_parse = tp1 - tp0
        if not nnA or not nnB:
            continue

        # ---- compare stage (L2 heavy) ----
        tc0 = time.perf_counter()
        compare_rows, summ = build_compare_rows(nnA, nnB, vocab)
        tc1 = time.perf_counter()
        t_compare = tc1 - tc0

        # ---- prematch stage ----
        tm0 = time.perf_counter()
        d_measure = to_measure(dataset[doc_id])
        plus0, minus0 = build_delta_masses(q_measure, d_measure)
        delta_total = float(sum(plus0.values()))
        match_edges, rem_plus, rem_minus, gstat = greedy_prematch(compare_rows, plus0, minus0)
        prem_flow = float(gstat["total_flow_sent"])
        prem_ratio = 0.0 if delta_total <= 0 else prem_flow / delta_total
        residual_mass = float(gstat["sum_rem_plus"])
        nz_plus = int(gstat["nonzero_rem_plus"])
        nz_minus = int(gstat["nonzero_rem_minus"])
        prem_c = prematch_cost(match_edges)
        tm1 = time.perf_counter()
        t_prematch = tm1 - tm0

        # ---- costs stage ----
        t_costA_base = t_costA_res = t_costB_base = t_costB_res = 0.0
        baseline_A = residual_A = total_A = improve_A = ""
        baseline_B = residual_B = total_B = improve_B = ""

        if not args.skip_costs:
            # seedA baseline
            tcb0 = time.perf_counter()
            baseline_A = residual_flowtree_cost(args.so_dir, vocab, hdrA.seed, plus0, minus0)
            tcb1 = time.perf_counter()
            t_costA_base = tcb1 - tcb0

            # seedA residual
            tcr0 = time.perf_counter()
            residual_A = residual_flowtree_cost(args.so_dir, vocab, hdrA.seed, rem_plus, rem_minus)
            tcr1 = time.perf_counter()
            t_costA_res = tcr1 - tcr0

            total_A = float(prem_c + float(residual_A))
            improve_A = float(float(baseline_A) - total_A)

            # seedB baseline
            tdb0 = time.perf_counter()
            baseline_B = residual_flowtree_cost(args.so_dir, vocab, hdrB.seed, plus0, minus0)
            tdb1 = time.perf_counter()
            t_costB_base = tdb1 - tdb0

            # seedB residual
            tdr0 = time.perf_counter()
            residual_B = residual_flowtree_cost(args.so_dir, vocab, hdrB.seed, rem_plus, rem_minus)
            tdr1 = time.perf_counter()
            t_costB_res = tdr1 - tdr0

            total_B = float(prem_c + float(residual_B))
            improve_B = float(float(baseline_B) - total_B)

        t1 = time.perf_counter()
        t_total = t1 - t0

        rec: Dict[str, Any] = {
            "qid": args.qid,
            "doc_id": doc_id,
            "seedA": hdrA.seed,
            "seedB": hdrB.seed,
            "packedN_A": hdrA.packedN,
            "packedN_B": hdrB.packedN,
            "common_u": summ["common_u"],
            "same_nn": summ["same_nn"],
            "diff_nn": summ["diff_nn"],
            "picked_A": summ["picked_A"],
            "picked_B": summ["picked_B"],
            "avg_abs_l2_gap": summ["avg_abs_l2_gap"],
            "delta_total": delta_total,
            "prematched_flow": prem_flow,
            "prematch_ratio": prem_ratio,
            "residual_mass": residual_mass,
            "nz_plus": nz_plus,
            "nz_minus": nz_minus,
            "prematch_cost": prem_c,
            "baseline_A": baseline_A,
            "residual_A": residual_A,
            "total_A": total_A,
            "improve_A": improve_A,
            "baseline_B": baseline_B,
            "residual_B": residual_B,
            "total_B": total_B,
            "improve_B": improve_B,
            "t_parse": t_parse,
            "t_compare": t_compare,
            "t_prematch": t_prematch,
            "t_costA_base": t_costA_base,
            "t_costA_res": t_costA_res,
            "t_costB_base": t_costB_base,
            "t_costB_res": t_costB_res,
            "t_total": t_total,
        }
        rows_csv.append(rec)

        # accumulate times
        T_parse.append(t_parse)
        T_compare.append(t_compare)
        T_prematch.append(t_prematch)
        T_costA_base.append(t_costA_base)
        T_costA_res.append(t_costA_res)
        T_costB_base.append(t_costB_base)
        T_costB_res.append(t_costB_res)
        T_total.append(t_total)

    # write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    print(f"[done] wrote {len(rows_csv)} rows -> {args.out_csv}")
    print()

    # averages
    n = len(rows_csv)
    if n == 0:
        return

    print("[timing avg over docs]")
    print(f"  avg t_parse    = {mean(T_parse):.6f} s")
    print(f"  avg t_compare  = {mean(T_compare):.6f} s")
    print(f"  avg t_prematch = {mean(T_prematch):.6f} s")
    if not args.skip_costs:
        print(f"  avg t_costA_base = {mean(T_costA_base):.6f} s")
        print(f"  avg t_costA_res  = {mean(T_costA_res):.6f} s")
        print(f"  avg t_costB_base = {mean(T_costB_base):.6f} s")
        print(f"  avg t_costB_res  = {mean(T_costB_res):.6f} s")
        print(f"  avg t_cost_total = {(mean(T_costA_base)+mean(T_costA_res)+mean(T_costB_base)+mean(T_costB_res)):.6f} s")
    print(f"  avg t_total    = {mean(T_total):.6f} s")

    if args.skip_costs:
        return

    # show top improvements
    def _f(x):
        try:
            return float(x)
        except Exception:
            return -1e18

    topA = sorted(rows_csv, key=lambda r: _f(r["improve_A"]), reverse=True)[: args.topk]
    topB = sorted(rows_csv, key=lambda r: _f(r["improve_B"]), reverse=True)[: args.topk]

    print()
    print("[top improvement by seedA]")
    for r in topA:
        print(f"  doc={r['doc_id']:>3}  improve_A={float(r['improve_A']): .6f}  prem_ratio={float(r['prematch_ratio']):.3f}  delta={float(r['delta_total']):.6f}  nz(+,-)=({r['nz_plus']},{r['nz_minus']})")

    print()
    print("[top improvement by seedB]")
    for r in topB:
        print(f"  doc={r['doc_id']:>3}  improve_B={float(r['improve_B']): .6f}  prem_ratio={float(r['prematch_ratio']):.3f}  delta={float(r['delta_total']):.6f}  nz(+,-)=({r['nz_plus']},{r['nz_minus']})")


if __name__ == "__main__":
    main()