#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_two_logs_pick_greedy_0313.py  (timed version)

追加したいこと:
  - どのフェーズで時間がかかっているかを計測して表示できるようにする
  - 計測は time.perf_counter() を使う
  - 「seedごとに毎回 load_vocabulary する」みたいなノイズを避けるため、
    C++ OTEstimators は seedA/seedB それぞれ1回だけ初期化して使い回す
    (baseline/residual の2回呼び出しで同じ ot インスタンスを使う)

出力:
  - 既存の summary / flow stats / cost blocks は維持
  - timing はデフォルトで summary を表示
    * --timing off      : timing表示なし
    * --timing summary  : 主要フェーズのみ
    * --timing full     : seedA/seedB の init/各costの内訳まで出す
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# ============================================================
# Timing helpers
# ============================================================

class Timers:
    def __init__(self) -> None:
        self.t: Dict[str, float] = {}

    def add(self, key: str, dt: float) -> None:
        self.t[key] = self.t.get(key, 0.0) + float(dt)

    def span(self, key: str):
        return _TimerSpan(self, key)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.t.get(key, default))

    def items(self) -> List[Tuple[str, float]]:
        return sorted(self.t.items(), key=lambda kv: kv[0])


class _TimerSpan:
    def __init__(self, timers: Timers, key: str) -> None:
        self.timers = timers
        self.key = key
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        self.timers.add(self.key, t1 - self.t0)
        return False


def print_timing(timers: Timers, mode: str) -> None:
    if mode == "off":
        return

    def p(label: str, key: str):
        print(f"  {label:<18s}= {timers.get(key):.6f} s")

    print("[timing]")
    if mode == "summary":
        p("t_total", "t_total")
        p("t_load_data", "t_load_data")
        p("t_read_logs", "t_read_logs")
        p("t_parse_doc", "t_parse_doc")
        p("t_compare", "t_compare")
        p("t_prematch", "t_prematch")
        p("t_cpp_total", "t_cpp_total")
        return

    # full
    p("t_total", "t_total")
    print("  ---- python ----")
    p("t_load_data", "t_load_data")
    p("t_read_logs", "t_read_logs")
    p("t_parse_doc", "t_parse_doc")
    p("t_parse_rows", "t_parse_rows")
    p("t_compare", "t_compare")
    p("t_prematch", "t_prematch")
    print("  ---- c++/pybind ----")
    p("t_import_so", "t_import_so")
    p("t_init_A", "t_init_A")
    p("t_init_B", "t_init_B")
    p("t_costA_base", "t_costA_base")
    p("t_costA_res", "t_costA_res")
    p("t_costB_base", "t_costB_base")
    p("t_costB_res", "t_costB_res")
    p("t_cpp_total", "t_cpp_total")


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
    seed: Optional[int]
    packedN: Optional[int]
    H: Optional[int]


@dataclass
class CompareRow:
    u_id: int
    A: NNRow
    l2_A: float
    B: NNRow
    l2_B: float
    pick: str           # "A" or "B"
    picked_row: NNRow
    picked_l2: float
    reason: str
    ord_priority: int = -1


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
# Utility loaders / parsers
# ============================================================

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    ids = [int(x) for x in ids]
    if len(ids) == 0:
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
    return sorted([(vid, w / s) for vid, w in acc.items()], key=lambda x: x[0])


def to_measure(obj: Any) -> List[Tuple[int, float]]:
    """
    1つの測度（query/doc）を list[(vid, weight)] に変換する。
    入力形式を柔軟に吸収:
      - list[int] / ndarray(1D): IDのみ → 一様重み（正規化）
      - list[(id,w)] / ndarray(N,2): そのまま正規化
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
            out: List[Tuple[int, float]] = []
            for row in obj:
                out.append((int(row[0]), float(row[1])))
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


def load_measures(data_dir: str, qid: int, doc_id: int):
    queries = np.load(os.path.join(data_dir, "queries.npy"), allow_pickle=True)
    dataset = np.load(os.path.join(data_dir, "dataset.npy"), allow_pickle=True)
    vocab = np.load(os.path.join(data_dir, "vocab.npy"), mmap_mode="r")

    q_raw = queries[qid]
    d_raw = dataset[doc_id]

    q = to_measure(q_raw)
    d = to_measure(d_raw)

    return q, d, vocab


# ============================================================
# Parse NN log (flowtree_real_1d_nn_pipeline.py output)
# ============================================================

_HEADER_RE = re.compile(
    r"^===\s+doc_id=(?P<doc_id>\d+)\s+idx=(?P<idx>\d+)\s+dump_seed=(?P<seed>-?\d+)\s+packedN=(?P<packedN>\d+)\s+H=(?P<H>\d+)\s+===\s*$",
    re.M,
)

_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)


def extract_doc_block(text: str, doc_id: int) -> Tuple[DocHeader, str]:
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError("No doc header found in log.")

    for i, m in enumerate(headers):
        did = int(m.group("doc_id"))
        if did != doc_id:
            continue

        start = m.start()
        end = headers[i + 1].start() if (i + 1) < len(headers) else len(text)
        block = text[start:end]

        hdr = DocHeader(
            doc_id=did,
            idx=int(m.group("idx")),
            seed=int(m.group("seed")),
            packedN=int(m.group("packedN")),
            H=int(m.group("H")),
        )
        return hdr, block

    raise ValueError(f"doc_id={doc_id} block not found in log.")


def parse_nn_table_from_block(block: str) -> Dict[int, NNRow]:
    rows = _ROW_RE.findall(block)
    if not rows:
        raise ValueError("No NN rows found in the selected doc block.")

    out: Dict[int, NNRow] = {}
    for u_id, u_x, v_id, v_x, d1 in rows:
        u = int(u_id)
        out[u] = NNRow(
            u_id=u,
            u_x=int(u_x),
            v_id=int(v_id),
            v_x=int(v_x),
            d1=int(d1),
        )
    return out


# ============================================================
# Math helpers
# ============================================================

def l2_dist(vocab: np.ndarray, u_id: int, v_id: int) -> float:
    a = vocab[int(u_id)]
    b = vocab[int(v_id)]
    return float(np.linalg.norm(a - b))


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


# ============================================================
# difference-mass helpers (FlowTree-consistent)
# ============================================================

def build_delta_masses(
    q_measure: List[Tuple[int, float]],
    d_measure: List[Tuple[int, float]],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    FlowTree と整合する差分質量:
      delta = q - d
      plus  = max(delta, 0)
      minus = max(-delta, 0)
    """
    qd = measure_to_dict(q_measure)
    dd = measure_to_dict(d_measure)

    vids = sorted(set(qd.keys()) | set(dd.keys()))
    plus_mass: Dict[int, float] = {}
    minus_mass: Dict[int, float] = {}
    delta_map: Dict[int, float] = {}

    eps = 1e-15
    for vid in vids:
        q = float(qd.get(vid, 0.0))
        d = float(dd.get(vid, 0.0))
        delta = q - d
        delta_map[int(vid)] = delta
        if delta > eps:
            plus_mass[int(vid)] = delta
        elif delta < -eps:
            minus_mass[int(vid)] = -delta

    return plus_mass, minus_mass, delta_map


# ============================================================
# Core: compare two logs and greedy prematch
# ============================================================

def build_compare_rows(
    nnA: Dict[int, NNRow],
    nnB: Dict[int, NNRow],
    vocab: np.ndarray,
    priority: str = "A",
) -> Tuple[List[CompareRow], Dict[str, Any]]:
    uA = set(nnA.keys())
    uB = set(nnB.keys())
    common = sorted(uA & uB)
    onlyA = sorted(uA - uB)
    onlyB = sorted(uB - uA)

    rows: List[CompareRow] = []
    picked_A = 0
    picked_B = 0
    same_nn = 0
    diff_nn = 0
    l2_gap_sum = 0.0

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
            reason = "L2_A <= L2_B" if abs(l2A - l2B) < 1e-15 else "L2_A < L2_B"
            picked_A += 1
        else:
            pick = "B"
            picked = rB
            picked_l2 = l2B
            reason = "L2_B < L2_A"
            picked_B += 1

        rows.append(
            CompareRow(
                u_id=u,
                A=rA, l2_A=l2A,
                B=rB, l2_B=l2B,
                pick=pick,
                picked_row=picked,
                picked_l2=picked_l2,
                reason=reason,
            )
        )

    if priority != "A":
        raise ValueError("This script currently supports priority='A' only.")
    rows.sort(key=lambda r: (r.A.d1, r.A.u_x, r.u_id))
    for i, r in enumerate(rows, start=1):
        r.ord_priority = i

    summary = {
        "common_u": len(common),
        "onlyA_u": len(onlyA),
        "onlyB_u": len(onlyB),
        "same_nn": same_nn,
        "diff_nn": diff_nn,
        "picked_A": picked_A,
        "picked_B": picked_B,
        "avg_abs_l2_gap": (l2_gap_sum / len(common)) if common else math.nan,
        "priority": "A",
    }
    return rows, summary


def greedy_prematch_delta_mass(
    compare_rows: List[CompareRow],
    plus_mass_in: Dict[int, float],
    minus_mass_in: Dict[int, float],
) -> Tuple[List[MatchEdge], Dict[int, float], Dict[int, float], Dict[str, Any]]:
    rem_plus = dict(plus_mass_in)
    rem_minus = dict(minus_mass_in)

    matched: List[MatchEdge] = []
    skipped_u_exhausted = 0
    skipped_v_exhausted = 0
    skipped_missing_u = 0
    skipped_missing_v = 0
    total_flow = 0.0

    for r in compare_rows:
        u = int(r.u_id)
        v = int(r.picked_row.v_id)

        mu = float(rem_plus.get(u, 0.0))
        mv = float(rem_minus.get(v, 0.0))

        if u not in rem_plus:
            skipped_missing_u += 1
            continue
        if v not in rem_minus:
            skipped_missing_v += 1
            continue
        if mu <= 0.0:
            skipped_u_exhausted += 1
            continue
        if mv <= 0.0:
            skipped_v_exhausted += 1
            continue

        f = min(mu, mv)
        rem_plus[u] = mu - f
        rem_minus[v] = mv - f
        total_flow += f

        matched.append(
            MatchEdge(
                ord_priority=r.ord_priority,
                u_id=u,
                pick=r.pick,
                v_id=v,
                d1_pick=int(r.picked_row.d1),
                l2_pick=float(r.picked_l2),
                flow=float(f),
                rem_u=float(rem_plus[u]),
                rem_v=float(rem_minus[v]),
            )
        )

    eps = 1e-15
    sum_rem_plus = sum(x for x in rem_plus.values() if x > eps)
    sum_rem_minus = sum(x for x in rem_minus.values() if x > eps)
    nonzero_plus = sum(1 for x in rem_plus.values() if x > eps)
    nonzero_minus = sum(1 for x in rem_minus.values() if x > eps)

    stats = {
        "cand_total": len(compare_rows),
        "matched_edges": len(matched),
        "skipped_u_exhausted": skipped_u_exhausted,
        "skipped_v_exhausted": skipped_v_exhausted,
        "skipped_missing_u": skipped_missing_u,
        "skipped_missing_v": skipped_missing_v,
        "total_flow_sent": float(total_flow),
        "sum_rem_plus": float(sum_rem_plus),
        "sum_rem_minus": float(sum_rem_minus),
        "nonzero_rem_plus": int(nonzero_plus),
        "nonzero_rem_minus": int(nonzero_minus),
    }
    return matched, rem_plus, rem_minus, stats


def prematch_cost_from_edges(match_edges: List[MatchEdge]) -> float:
    return float(sum(e.l2_pick * e.flow for e in match_edges))


# ============================================================
# C++ FlowTree calls (seedA/seedB reuse)
# ============================================================

def import_ot_estimators(so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    import ot_estimators  # type: ignore
    return ot_estimators


def prepare_pair_mode_if_possible(ot) -> None:
    # pair mode helper exists in your modified C++ in some iterations
    if hasattr(ot, "prepare_pair_mode"):
        ot.prepare_pair_mode()
        return
    # fallback: stage=2 にするためのダミー dataset
    dummy = [[(0, 1.0)]]
    ot.load_dataset(dummy)


def _balance_measures(
    a: List[Tuple[int, float]],
    b: List[Tuple[int, float]],
    balance_eps: float = 1e-9,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    sum(a) == sum(b) になるように、数値誤差分だけスケールして合わせる。
    """
    s1 = sum_measure(a)
    s2 = sum_measure(b)
    if s1 <= 0.0 or s2 <= 0.0:
        return a, b
    if abs(s1 - s2) <= balance_eps:
        return a, b
    if s1 > s2:
        scale = s2 / s1
        a = [(vid, w * scale) for vid, w in a]
    else:
        scale = s1 / s2
        b = [(vid, w * scale) for vid, w in b]
    return a, b


def build_ot_for_seed(so_dir: str, vocab: np.ndarray, seed: int, timers: Timers, key_prefix: str):
    """
    seedごとに OTEstimators を1回だけ作って load_vocabulary まで済ませる。
    """
    with timers.span("t_import_so"):
        ot_estimators = import_ot_estimators(so_dir)

    ot = ot_estimators.OTEstimators()
    with timers.span(f"t_init_{key_prefix}"):
        ot.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seed))
        prepare_pair_mode_if_possible(ot)

    if not hasattr(ot, "flowtree_distance_pair"):
        raise RuntimeError("ot_estimators.OTEstimators has no flowtree_distance_pair (C++側に未実装/未export)")
    return ot


def flowtree_cost_pair(
    ot,
    plus_mass: Dict[int, float],
    minus_mass: Dict[int, float],
    balance_eps: float = 1e-9,
) -> float:
    plus = dict_to_measure(plus_mass)
    minus = dict_to_measure(minus_mass)
    s1 = sum_measure(plus)
    s2 = sum_measure(minus)
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0
    plus, minus = _balance_measures(plus, minus, balance_eps=balance_eps)
    return float(ot.flowtree_distance_pair(plus, minus))


# ============================================================
# Printing (row-heavy prints are gated)
# ============================================================

def print_summary(
    hdrA: DocHeader,
    hdrB: DocHeader,
    compare_summary: Dict[str, Any],
    logA: str,
    logB: str,
    doc_id: int,
):
    print(f"[input] logA={logA}")
    print(f"[input] logB={logB}")
    print(f"[input] doc_id={doc_id}")
    print()
    print("[summary]")
    print(f"  logA: doc_id={hdrA.doc_id} idx={hdrA.idx} seed={hdrA.seed} packedN={hdrA.packedN} H={hdrA.H}")
    print(f"  logB: doc_id={hdrB.doc_id} idx={hdrB.idx} seed={hdrB.seed} packedN={hdrB.packedN} H={hdrB.H}")
    print(f"  priority = {compare_summary['priority']} (sort by d1 on that log)")
    print(f"  common_u = {compare_summary['common_u']}")
    print(f"  onlyA_u  = {compare_summary['onlyA_u']}")
    print(f"  onlyB_u  = {compare_summary['onlyB_u']}")
    print(f"  same_NN(v_id_A==v_id_B) = {compare_summary['same_nn']}")
    print(f"  diff_NN(v_id_A!=v_id_B) = {compare_summary['diff_nn']}")
    print(f"  picked A = {compare_summary['picked_A']}")
    print(f"  picked B = {compare_summary['picked_B']}")
    print(f"  avg |L2_A-L2_B| = {compare_summary['avg_abs_l2_gap']:.6f}")
    print()


def print_compare_table(rows: List[CompareRow], limit: int = 200):
    print("[table] u-keyed compare & pick (priority by seedA d1 asc)")
    print(" ord |  u_id |  u_x(A)  v_id(A)  v_x(A)  d1(A)      L2(A)  ||  u_x(B)  v_id(B)  v_x(B)  d1(B)      L2(B)  || pick  picked_v  picked_d1   picked_L2  reason")
    print("-" * 190)
    for r in rows[:limit]:
        print(
            f"{r.ord_priority:4d} |"
            f" {r.u_id:5d} |"
            f" {r.A.u_x:7d} {r.A.v_id:8d} {r.A.v_x:7d} {r.A.d1:7d} {r.l2_A:10.6f}  ||"
            f" {r.B.u_x:7d} {r.B.v_id:8d} {r.B.v_x:7d} {r.B.d1:7d} {r.l2_B:10.6f}  ||"
            f" {r.pick:>3s} {r.picked_row.v_id:9d} {r.picked_row.d1:10d} {r.picked_l2:11.6f}  {r.reason}"
        )
    print()


def print_picked_table(rows: List[CompareRow], limit: int = 200):
    print("[table] picked candidate only (for next greedy step)")
    print(" ord   u_id   pick   picked_v   picked_d1   picked_L2")
    print("-" * 58)
    for r in rows[:limit]:
        print(
            f"{r.ord_priority:4d} {r.u_id:6d} {r.pick:>6s} {r.picked_row.v_id:10d} {r.picked_row.d1:11d} {r.picked_l2:11.6f}"
        )
    print()


def print_greedy_rows(match_edges: List[MatchEdge], show_top: int = 200):
    print("[matched] greedy flows (show_top={})".format(show_top))
    print(" ord   u_vid   pick      v_vid   picked_d1      L2        flow       rem_u       rem_v")
    print("-" * 92)
    for e in match_edges[:show_top]:
        print(
            f"{e.ord_priority:4d} {e.u_id:7d} {e.pick:>6s} {e.v_id:10d} {e.d1_pick:10d} "
            f"{e.l2_pick:8.6f} {e.flow:10.6f} {e.rem_u:10.6f} {e.rem_v:10.6f}"
        )
    print()


def print_flow_stats(delta_total: float, prematched: float, residual: float, nz_plus: int, nz_minus: int):
    ratio = 0.0 if delta_total <= 0 else prematched / delta_total
    print("[prematch flow]")
    print(f"  delta_total (sum plus/minus) = {delta_total:.6f}")
    print(f"  prematched_flow_sent         = {prematched:.6f}   (ratio={ratio:.3f})")
    print(f"  residual_mass                = {residual:.6f}   (#nonzero plus={nz_plus}, minus={nz_minus})")
    print()


def print_cost_block(seed_label: str, seed_value: int, baseline: float, prem_cost: float, residual: float):
    total = prem_cost + residual
    improvement = baseline - total
    print(f"[cost {seed_label}] (seed={seed_value})")
    print(f"  baseline_flowtree_cost     = {baseline:.6f}")
    print(f"  prematch_cost(L2*flow)     = {prem_cost:.6f}")
    print(f"  residual_flowtree_cost     = {residual:.6f}")
    print(f"  total_cost_with_prematch   = {total:.6f}")
    print(f"  improvement (baseline-total)= {improvement:.6f}")
    print()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing vocab.npy / queries.npy / dataset.npy")
    ap.add_argument("--so_dir", required=True, help="dir containing ot_estimators*.so")
    ap.add_argument("--logA", required=True, help="NN log for seed A")
    ap.add_argument("--logB", required=True, help="NN log for seed B")
    ap.add_argument("--doc_id", type=int, required=True, help="document id to inspect")
    ap.add_argument("--qid", type=int, default=0, help="query id (default: 0)")

    # seeds control: default both
    ap.add_argument("--seeds", choices=["A", "B", "AB"], default="AB",
                    help="which seeds to compute costs for. default: AB")

    # timing
    ap.add_argument("--timing", choices=["off", "summary", "full"], default="summary",
                    help="timing output level. default: summary")

    # table controls
    ap.add_argument("--show_compare", type=int, default=200, help="rows to show in compare/picked tables (when --show_tables)")
    ap.add_argument("--show_match", type=int, default=200, help="rows to show in matched table (when --show_match_rows)")
    ap.add_argument("--show_tables", action="store_true",
                    help="print row-based tables (compare/picked). default: off")
    ap.add_argument("--show_match_rows", action="store_true",
                    help="print matched rows table. default: off")
    ap.add_argument("--no_compare_table", action="store_true",
                    help="do not print compare table (only relevant with --show_tables)")
    ap.add_argument("--no_picked_table", action="store_true",
                    help="do not print picked-only table (only relevant with --show_tables)")

    args = ap.parse_args()

    timers = Timers()
    t0_total = time.perf_counter()

    # Load data
    with timers.span("t_load_data"):
        q_measure, d_measure, vocab = load_measures(args.data_dir, qid=args.qid, doc_id=args.doc_id)

    # Read logs
    with timers.span("t_read_logs"):
        textA = read_text(args.logA)
        textB = read_text(args.logB)

    # Extract blocks + parse rows
    with timers.span("t_parse_doc"):
        hdrA, blockA = extract_doc_block(textA, args.doc_id)
        hdrB, blockB = extract_doc_block(textB, args.doc_id)

    with timers.span("t_parse_rows"):
        nnA = parse_nn_table_from_block(blockA)
        nnB = parse_nn_table_from_block(blockB)

    # Compare and pick
    with timers.span("t_compare"):
        rows, compare_summary = build_compare_rows(nnA, nnB, vocab, priority="A")

    # Base delta masses
    plus0, minus0, _ = build_delta_masses(q_measure, d_measure)
    delta_total = float(sum(plus0.values()))  # == sum(minus0.values()) ideally

    # Prematch
    with timers.span("t_prematch"):
        match_edges, rem_plus, rem_minus, greedy_stats = greedy_prematch_delta_mass(rows, plus0, minus0)
        prem_cost = prematch_cost_from_edges(match_edges)
        residual_mass = float(greedy_stats["sum_rem_plus"])  # == sum_rem_minus
        prematched_flow = float(greedy_stats["total_flow_sent"])

    # Print summary (always)
    print_summary(hdrA, hdrB, compare_summary, args.logA, args.logB, args.doc_id)

    # Row-heavy outputs (default off)
    if args.show_tables:
        if not args.no_compare_table:
            print_compare_table(rows, limit=args.show_compare)
        if not args.no_picked_table:
            print_picked_table(rows, limit=args.show_compare)
    if args.show_match_rows:
        print_greedy_rows(match_edges, show_top=args.show_match)

    # Flow stats (default ON)
    print_flow_stats(
        delta_total=delta_total,
        prematched=prematched_flow,
        residual=residual_mass,
        nz_plus=int(greedy_stats["nonzero_rem_plus"]),
        nz_minus=int(greedy_stats["nonzero_rem_minus"]),
    )

    # Costs for requested seeds (default both)
    seeds_to_run: List[Tuple[str, int]] = []
    if args.seeds in ("A", "AB"):
        if hdrA.seed is None:
            raise RuntimeError("hdrA.seed is None (log header parse failed?)")
        seeds_to_run.append(("A", int(hdrA.seed)))
    if args.seeds in ("B", "AB"):
        if hdrB.seed is None:
            raise RuntimeError("hdrB.seed is None (log header parse failed?)")
        seeds_to_run.append(("B", int(hdrB.seed)))

    # Build ot instances once per seed (this is important for stable timing)
    otA = None
    otB = None

    with timers.span("t_cpp_total"):
        if any(lbl == "A" for lbl, _ in seeds_to_run):
            otA = build_ot_for_seed(args.so_dir, vocab, dict(seeds_to_run)["A"], timers, "A")
        if any(lbl == "B" for lbl, _ in seeds_to_run):
            otB = build_ot_for_seed(args.so_dir, vocab, dict(seeds_to_run)["B"], timers, "B")

        for label, seed_value in seeds_to_run:
            if label == "A":
                assert otA is not None
                with timers.span("t_costA_base"):
                    baseline = flowtree_cost_pair(otA, plus0, minus0)
                with timers.span("t_costA_res"):
                    residual = flowtree_cost_pair(otA, rem_plus, rem_minus)
                print_cost_block(label, seed_value, baseline, prem_cost, residual)

            elif label == "B":
                assert otB is not None
                with timers.span("t_costB_base"):
                    baseline = flowtree_cost_pair(otB, plus0, minus0)
                with timers.span("t_costB_res"):
                    residual = flowtree_cost_pair(otB, rem_plus, rem_minus)
                print_cost_block(label, seed_value, baseline, prem_cost, residual)

    timers.add("t_total", time.perf_counter() - t0_total)

    # Timing print (default summary)
    print_timing(timers, args.timing)


if __name__ == "__main__":
    main()