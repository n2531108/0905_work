#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_two_logs_pick_greedy_0224.py

目的:
  - seed違いの2本の木から得た NN ログ (u -> v, d1) を読み取る
  - 各 u について A/B の候補を比較し、L2 が小さい方を採用
  - ただし処理順は「木Aの d1 昇順」に固定
  - 採用候補に対して greedy にプレマッチングを行う
  - どこをマッチングしたか（flow, 残量）を表示

前提:
  - queries.npy / dataset.npy は object配列（list[int] など）でもOK
  - 重みが無い場合は一様重みを仮定
  - NNログは flowtree_real_1d_nn_pipeline.py の出力形式
    （"=== doc_id=... ===" と "[table] ...", 行形式 u_id u_x v_id v_x d1）

使い方例:
  python3 nn_two_logs_pick_greedy_0224.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --logA "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed110.log" \
    --logB "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed111.log" \
    --doc_id 0 \
    --qid 0 \
    --show_compare 200 \
    --show_match 200
"""

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

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
    seed: Optional[int]
    packedN: Optional[int]
    H: Optional[int]


@dataclass
class CompareRow:
    # key
    u_id: int

    # A row
    A: NNRow
    l2_A: float

    # B row
    B: NNRow
    l2_B: float

    # picked by smaller L2
    pick: str           # "A" or "B"
    picked_row: NNRow
    picked_l2: float
    reason: str

    # order priority uses A.d1 only (fixed policy)
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
      - list[int] / ndarray(1D): IDのみ → 一様重み
      - list[(id,w)] / ndarray(N,2): そのまま正規化
    """
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
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
    diff = a - b
    return float(np.linalg.norm(diff))


def measure_to_dict(m: List[Tuple[int, float]]) -> Dict[int, float]:
    return {int(vid): float(w) for vid, w in m}


# ============================================================
# [MOD] difference-mass helpers
# ============================================================

def build_delta_masses(
    q_measure: List[Tuple[int, float]],
    d_measure: List[Tuple[int, float]],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    [MOD]
    FlowTree と整合するように、元の query/doc 測度から差分質量を作る。

    Returns:
      plus_mass : vid -> max(q_mass - d_mass, 0)
      minus_mass: vid -> max(d_mass - q_mass, 0)
      delta_map : vid -> q_mass - d_mass
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


def greedy_prematch(
    compare_rows: List[CompareRow],
    q_measure: List[Tuple[int, float]],
    d_measure: List[Tuple[int, float]],
) -> Tuple[List[MatchEdge], Dict[int, float], Dict[int, float], Dict[str, Any]]:
    """
    [MOD]
    compare_rows は ord_priority で既にソート済み（= 木Aの d1順）
    picked_row を使って greedy に流す。

    ただし流す質量は元の query/doc 重みではなく、
    FlowTree と整合する差分質量:
      rem_plus[u]  = max(q_u - d_u, 0)
      rem_minus[v] = max(d_v - q_v, 0)
    を使う。
    """
    rem_plus, rem_minus, delta_map = build_delta_masses(q_measure, d_measure)

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
        "rows_used": len(matched),
        "skipped_u_exhausted": skipped_u_exhausted,
        "skipped_v_exhausted": skipped_v_exhausted,
        "skipped_missing_u": skipped_missing_u,
        "skipped_missing_v": skipped_missing_v,
        "total_flow_sent": total_flow,
        "sum_rem_plus": sum_rem_plus,
        "sum_rem_minus": sum_rem_minus,
        "nonzero_rem_plus": nonzero_plus,
        "nonzero_rem_minus": nonzero_minus,
        # [MOD] 参考: 差分残差の総量
        "sum_delta_pos": sum(float(x) for x in rem_plus.values()),
        "sum_delta_neg": sum(float(x) for x in rem_minus.values()),
    }
    return matched, rem_plus, rem_minus, stats


# ============================================================
# Printing
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
    print(f"  (priority=A) rows where B selected = {compare_summary['picked_B']}")
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


def print_greedy_result(match_edges: List[MatchEdge], stats: Dict[str, Any], show_top: int = 200):
    print("[greedy] prematch result (priority = seedA d1 asc, candidate picked by min L2(A,B))")
    print(f"[stats] cand_total={stats['cand_total']}  matched_edges={stats['matched_edges']}  rows_used={stats['rows_used']}")
    print(
        f"[stats] skipped(u_exhausted)={stats['skipped_u_exhausted']}  "
        f"skipped(v_exhausted)={stats['skipped_v_exhausted']}  "
        f"skipped(missing_u)={stats['skipped_missing_u']}  "
        f"skipped(missing_v)={stats['skipped_missing_v']}"
    )
    print(f"[summary] total_flow_sent = {stats['total_flow_sent']:.6f}")
    print()
    print("[matched] greedy flows (show_top={})".format(show_top))
    print(" ord   u_vid   pick      v_vid   picked_d1      L2        flow       rem_u       rem_v")
    print("-" * 92)
    for e in match_edges[:show_top]:
        print(
            f"{e.ord_priority:4d} {e.u_id:7d} {e.pick:>6s} {e.v_id:10d} {e.d1_pick:10d} "
            f"{e.l2_pick:8.6f} {e.flow:10.6f} {e.rem_u:10.6f} {e.rem_v:10.6f}"
        )
    print()
    print("[remaining residual]")
    print(f"  sum(rem_plus)  = {stats['sum_rem_plus']:.6f}   (#nonzero={stats['nonzero_rem_plus']})")
    print(f"  sum(rem_minus) = {stats['sum_rem_minus']:.6f}   (#nonzero={stats['nonzero_rem_minus']})")
    print()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing vocab.npy / queries.npy / dataset.npy")
    ap.add_argument("--logA", required=True, help="NN log for seed A")
    ap.add_argument("--logB", required=True, help="NN log for seed B")
    ap.add_argument("--doc_id", type=int, required=True, help="document id to inspect")
    ap.add_argument("--qid", type=int, default=0, help="query id (default: 0)")
    ap.add_argument("--show_compare", type=int, default=200, help="rows to show in compare table")
    ap.add_argument("--show_match", type=int, default=200, help="rows to show in greedy matched table")
    ap.add_argument("--no_compare_table", action="store_true", help="do not print detailed compare table")
    ap.add_argument("--no_picked_table", action="store_true", help="do not print picked-only table")
    args = ap.parse_args()

    q_measure, d_measure, vocab = load_measures(args.data_dir, qid=args.qid, doc_id=args.doc_id)

    textA = read_text(args.logA)
    textB = read_text(args.logB)

    hdrA, blockA = extract_doc_block(textA, args.doc_id)
    hdrB, blockB = extract_doc_block(textB, args.doc_id)

    nnA = parse_nn_table_from_block(blockA)
    nnB = parse_nn_table_from_block(blockB)

    rows, compare_summary = build_compare_rows(nnA, nnB, vocab, priority="A")

    match_edges, rem_plus, rem_minus, greedy_stats = greedy_prematch(rows, q_measure, d_measure)

    print_summary(hdrA, hdrB, compare_summary, args.logA, args.logB, args.doc_id)

    if not args.no_compare_table:
        print_compare_table(rows, limit=args.show_compare)

    if not args.no_picked_table:
        print_picked_table(rows, limit=args.show_compare)

    print_greedy_result(match_edges, greedy_stats, show_top=args.show_match)


if __name__ == "__main__":
    main()