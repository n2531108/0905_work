#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flowtree_two_tree_pipeline.py

目的:
  2本の木 (seedA / seedB) を用いて、単一の query-doc ペアに対し

    packed subtree 取得
      -> Algorithm A (1D tree embedding)
      -> delta による +/- split
      -> Algorithm B (+ -> - 1D nearest neighbor)
      -> A/B 候補を元空間 L2 で比較
      -> greedy prematch
      -> residual を FlowTree に渡してコスト評価

  までを、dump やログを使わずに一気通貫で行う。

前提:
  ot_estimators_twotree が build 済みであり、少なくとも次が見えていること:
    - load_vocabulary(...)
    - build_packed_subtree(...)
    - get_last_packed_root()
    - get_last_packed_edges()
    - get_last_packed_is_leaf()
    - get_last_packed_unleaf()
    - get_last_packed_delta()
    - flowtree_distance_pair(...)

重要:
  本スクリプトは 2本の「異なる木」を作るため、C++ 側が
    load_vocabulary(vocab, seed)
  を受け取れることを優先的に期待する。

  もし現在の C++ 側が seed 引数をまだ受け取れない場合:
    - seedA == seedB のときだけ同一木として動作可能
    - seedA != seedB では真の二本木にならないため、明示的にエラーにする

使い方例:
  python3 flowtree_packed_1d_pipeline.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --so_dir   "/mnt/c/Users/成見/0905_work/native/build" \
    --qid 0 \
    --doc_id 0 \
    --seedA 110 \
    --seedB 111

比較表や prematch 行を出したい場合:
  python3 flowtree_packed_1d_pipeline.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --so_dir   "/mnt/c/Users/成見/0905_work/native/build" \
    --qid 0 \
    --doc_id 0 \
    --seedA 110 \
    --seedB 111 \
    --show_compare \
    --show_match

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ============================================================
# 1) データ構造
# ============================================================

@dataclass(frozen=True)
class Pt:
    id: int
    x: int


@dataclass(frozen=True)
class NNResult:
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int


@dataclass
class PackedSubtree:
    root: int
    edges: List[Tuple[int, int]]
    is_leaf: List[int]
    unleaf: List[int]
    delta: List[float]

    @property
    def n_nodes(self) -> int:
        return len(self.is_leaf)


@dataclass
class SingleTreeResult:
    seed: int
    packed: PackedSubtree
    height: int
    depth: List[int]
    leaf_x_by_node: Dict[int, int]
    plus_vids: List[int]
    minus_vids: List[int]
    sign_map: Dict[int, int]
    mass_map: Dict[int, float]
    nn_scan: Dict[int, NNResult]


@dataclass
class CompareRow:
    u_id: int
    A: NNResult
    l2_A: float
    B: NNResult
    l2_B: float
    pick: str
    picked_row: NNResult
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
# 2) measure ローダ
# ============================================================

# ids のリストを受け取って、全てに一様重みをつける関数。
def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    if len(ids) == 0:
        return []
    w = 1.0 / float(len(ids))
    return [(int(v), w) for v in ids]

# measure を正規化する関数。vid ごとに重みを合算し、全体で L1 正規化する。
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

# measure を list[(vid, weight)] の形に変換する関数。
def to_measure(obj: Any) -> List[Tuple[int, float]]:
    """
    1つの query/doc を list[(vid, weight)] に直す。
    想定:
      - [(vid, w), ...]
      - ndarray shape (L,2)
      - [vid, vid, ...] -> 一様重み
      - ndarray shape (L,) -> 一様重み
    """
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

# measure ローダ。ファイルから vocab の measure を読み込む。
def load_vocab(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise RuntimeError(f"missing vocab.npy: {path}")
    vocab = np.load(path, allow_pickle=False)
    if not isinstance(vocab, np.ndarray) or vocab.ndim != 2 or vocab.dtype != np.float32:
        raise RuntimeError(
            f"vocab.npy must be float32 2D, got type={type(vocab)} "
            f"ndim={getattr(vocab, 'ndim', None)} dtype={getattr(vocab, 'dtype', None)}"
        )
    return vocab

# measure ローダ。ファイルから query/dataset の measure を読み込む。to_measure を通す
def load_object_measures(path: str) -> List[List[Tuple[int, float]]]:
    if not os.path.exists(path):
        raise RuntimeError(f"missing: {path}")

    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [to_measure(x) for x in arr.tolist()]

    if isinstance(arr, np.ndarray) and arr.ndim >= 1:
        return [to_measure(arr[i]) for i in range(arr.shape[0])]

    return [to_measure(arr)]

#上記のローダをまとめたもの。vocab, dataset, queries を一気に読み込む。
def load_all_inputs(data_dir: str) -> Tuple[np.ndarray, List[List[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    vocab = load_vocab(os.path.join(data_dir, "vocab.npy"))
    dataset = load_object_measures(os.path.join(data_dir, "dataset.npy"))
    queries = load_object_measures(os.path.join(data_dir, "queries.npy"))
    return vocab, dataset, queries


# ============================================================
# 3) C++ 呼び出し
# ============================================================

def import_ot_estimators_twotree(so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    import ot_estimators_twotree  # type: ignore
    return ot_estimators_twotree


def load_vocabulary_with_seed(ot, vocab: np.ndarray, seed: int) -> None:
    """
    C++ 側が seed 引数つき load_vocabulary を持っていればそれを使う。
    なければ seed なし版にフォールバックするが、その場合は二本木を区別できない。
    """
    try:
        ot.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seed))
        return
    except TypeError:
        ot.load_vocabulary(np.asarray(vocab, dtype=np.float32))
        # seed なしのときは、呼び出し側で seedA != seedB を禁止する

# OTEstimators を二つ作り、それぞれに vocab をロードする。seedA, seedB を渡すが、C++ 側が受け取れない場合はエラーにする。
def build_two_ots(so_dir: str, vocab: np.ndarray, seedA: int, seedB: int):
    mod = import_ot_estimators_twotree(so_dir)

    otA = mod.OTEstimators()
    otB = mod.OTEstimators()

    # seed 付き API を試す
    seed_api_ok = True
    try:
        otA.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seedA))
        otB.load_vocabulary(np.asarray(vocab, dtype=np.float32), int(seedB))
    except TypeError:
        seed_api_ok = False
        if seedA != seedB:
            raise RuntimeError(
                "C++ 側の load_vocabulary が seed 引数を受け取れません。"
                "現在の C++ 実装では seedA != seedB の真の二本木は作れません。"
            )
        otA.load_vocabulary(np.asarray(vocab, dtype=np.float32))
        otB.load_vocabulary(np.asarray(vocab, dtype=np.float32))

    return otA, otB, seed_api_ok

# query と doc を C++ 側に渡して packed subtree を構築し、Python 側で PackedSubtree オブジェクトに変換して返す。
def build_packed_subtree_from_cpp(
    ot,
    query: List[Tuple[int, float]],
    doc: List[Tuple[int, float]],
) -> PackedSubtree:
    ot.build_packed_subtree(query, doc)

    root = int(ot.get_last_packed_root())
    edges = [(int(a), int(b)) for a, b in ot.get_last_packed_edges()]
    is_leaf = [int(x) for x in ot.get_last_packed_is_leaf()]
    unleaf = [int(x) for x in ot.get_last_packed_unleaf()]
    delta = [float(x) for x in ot.get_last_packed_delta()]

    if not (len(is_leaf) == len(unleaf) == len(delta)):
        raise RuntimeError(
            "packed subtree arrays have inconsistent lengths: "
            f"is_leaf={len(is_leaf)} unleaf={len(unleaf)} delta={len(delta)}"
        )

    return PackedSubtree(
        root=root,
        edges=edges,
        is_leaf=is_leaf,
        unleaf=unleaf,
        delta=delta,
    )


# ============================================================
# 4) packed subtree -> children
# ============================================================
# C++ 側からは root と edges のみが木構造を表す情報として得られるため、これを children のリストに変換する関数。
def children_from_edges(n_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    children = [[] for _ in range(n_nodes)]
    for p, q in edges:
        if 0 <= p < n_nodes and 0 <= q < n_nodes:
            children[p].append(q)
    for ch in children:
        ch.sort()
    return children


# ============================================================
# 5) Algorithm A
# ============================================================
# subtreeを一次元に埋め込むAlgorithm A を実装する関数。packed subtree の root と children を受け取り、葉 local node -> 1D座標 の辞書を返す。
def compute_depths(root: int, children: List[List[int]]) -> Tuple[List[int], int]:
    n = len(children)
    depth = [-1] * n
    q = deque([root])
    depth[root] = 0
    H = 0
    while q:
        u = q.popleft()
        H = max(H, depth[u])
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)
    return depth, H


def algoA_embed(
    root: int,
    children: List[List[int]],
    is_leaf: List[int],
) -> Tuple[Dict[int, int], int, List[int]]:
    """
    論文の Algorithm A にできるだけ忠実な形。
    葉 local node -> 1D座標 の辞書を返す。
    """
    n = len(children)
    f = {u: 0 for u in range(n)}
    depth, H = compute_depths(root, children)

    def add_on_leaves(u: int, delta: int) -> None:
        stack = [u]
        while stack:
            x = stack.pop()
            if is_leaf[x]:
                f[x] += delta
            else:
                for v in children[x]:
                    stack.append(v)

    def treeEmbed(u: int, k: int) -> int:
        if is_leaf[u]:
            f[u] = k
            return k
        i = 0
        ch = children[u]
        imax = len(ch) - 1
        delta_old = 0
        for c in ch:
            kprime = treeEmbed(c, k)
            delta_now = kprime - k
            k = kprime
            if i > 0:
                gap = max(delta_old, delta_now)
                add_on_leaves(c, gap)
                k += gap
            if i != imax:
                k += 1
            delta_old = delta_now
            i += 1
        return k

    treeEmbed(root, 0)

    reachable = [0] * n
    stack = [root]
    reachable[root] = 1
    while stack:
        u = stack.pop()
        for v in children[u]:
            if not reachable[v]:
                reachable[v] = 1
                stack.append(v)

    leaf_x = {u: f[u] for u in f if is_leaf[u] and reachable[u]}
    return leaf_x, H, depth


# ============================================================
# 6) delta による +/- split
# ============================================================
# Algorithm A の結果を受け取って、delta に基づいて葉を + と - に分割する関数。leaf_x_by_node は葉 local node -> 1D座標 の辞書、unleaf は葉でない local node -> vid のリスト、delta は local node ごとの delta のリスト。eps は delta をゼロとみなす閾値。
def split_by_delta(
    leaf_x_by_node: Dict[int, int],
    unleaf: List[int],
    delta: List[float],
    eps: float = 1e-12,
) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, float]]:
    sign_map: Dict[int, int] = {}
    mass_map: Dict[int, float] = {}

    plus_vids: List[int] = []
    minus_vids: List[int] = []

    for leaf_node in leaf_x_by_node.keys():
        if not (0 <= leaf_node < len(unleaf)):
            continue
        vid = int(unleaf[leaf_node])
        if vid < 0:
            continue
        d = float(delta[leaf_node])
        if abs(d) <= eps:
            continue
        sg = 1 if d > 0 else -1
        sign_map[vid] = sg
        mass_map[vid] = abs(d)
        if sg > 0:
            plus_vids.append(vid)
        else:
            minus_vids.append(vid)

    plus_vids.sort()
    minus_vids.sort()
    return plus_vids, minus_vids, sign_map, mass_map


# ============================================================
# 7) Algorithm B
# ============================================================
# ソート済みの一次元埋め込みを使って+ -> - 1D 最近傍を求める Algorithm B を実装する関数。Pt のリスト X, Y を受け取り、X の各点に対して Y の中の最近傍を NNResult として返す。
def sort_pts(pts: Iterable[Pt]) -> List[Pt]:
    return sorted(list(pts), key=lambda p: (p.x, p.id))


def abs_i(a: int) -> int:
    return a if a >= 0 else -a


def nn_scan_algoB_pseudocode(
    X: List[Pt],
    Y: List[Pt],
    tie_break: str = "left",
) -> List[NNResult]:
    if not Y:
        raise ValueError("Y が空なので最近傍が定義できません")
    if tie_break not in ("left", "right"):
        raise ValueError("tie_break は 'left' か 'right' を指定してください")

    Xs = sort_pts(X)
    Ys = sort_pts(Y)

    res: List[NNResult] = []
    j = 0
    m = len(Ys)

    for u in Xs:
        while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) < abs_i(Ys[j].x - u.x)):
            j += 1

        v = Ys[j]
        d = abs_i(v.x - u.x)

        if tie_break == "right":
            while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) == abs_i(Ys[j].x - u.x)):
                j += 1
                v = Ys[j]
                d = abs_i(v.x - u.x)

        res.append(NNResult(u.id, u.x, v.id, v.x, d))

    return res

# ============================================================
# 8) 単木結果
# ============================================================
#local_nodeではなくv_idに直す
def build_vid_to_x(
    leaf_x_by_node: Dict[int, int],
    unleaf: List[int],
) -> Dict[int, int]:
    vid_to_x: Dict[int, int] = {}
    for leaf_node, x in leaf_x_by_node.items():
        vid = int(unleaf[leaf_node])
        if vid >= 0:
            vid_to_x[vid] = int(x)
    return vid_to_x

#単木の全処理をまとめて行う関数。C++ 側に query と doc を渡して packed subtree を構築し、Algorithm A で葉を一次元に埋め込み、delta による +/- split を行い、Algorithm B で + -> - の最近傍を求めるまでを一気通貫で行う。結果は SingleTreeResult として返す。
def run_single_tree(
    ot,
    query: List[Tuple[int, float]],
    doc: List[Tuple[int, float]],
    seed: int,
    tie_break: str,
) -> SingleTreeResult:
    packed = build_packed_subtree_from_cpp(ot, query, doc)
    children = children_from_edges(packed.n_nodes, packed.edges)
    leaf_x_by_node, H, depth = algoA_embed(packed.root, children, packed.is_leaf)

    plus_vids, minus_vids, sign_map, mass_map = split_by_delta(
        leaf_x_by_node=leaf_x_by_node,
        unleaf=packed.unleaf,
        delta=packed.delta,
    )

    vid_to_x = build_vid_to_x(leaf_x_by_node, packed.unleaf)
    X_pts = [Pt(v, vid_to_x[v]) for v in plus_vids if v in vid_to_x]
    Y_pts = [Pt(v, vid_to_x[v]) for v in minus_vids if v in vid_to_x]

    nn_scan_list: List[NNResult] = []
    if X_pts and Y_pts:
        nn_scan_list = nn_scan_algoB_pseudocode(X_pts, Y_pts, tie_break=tie_break)

    nn_scan = {r.u_id: r for r in nn_scan_list}

    return SingleTreeResult(
        seed=seed,
        packed=packed,
        height=H,
        depth=depth,
        leaf_x_by_node=leaf_x_by_node,
        plus_vids=plus_vids,
        minus_vids=minus_vids,
        sign_map=sign_map,
        mass_map=mass_map,
        nn_scan=nn_scan,
    )


# ============================================================
# 9) 比較・prematch
# ============================================================

def l2_dist(vocab: np.ndarray, u_id: int, v_id: int) -> float:
    a = vocab[int(u_id)]
    b = vocab[int(v_id)]
    return float(np.linalg.norm(a - b))

#各uに対して二本の木の候補を元空間距離に応じて良いほうを採用する　また結果をcompare_rowsにまとめる
def build_compare_rows(
    nnA: Dict[int, NNResult],
    nnB: Dict[int, NNResult],
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
                A=rA,
                l2_A=l2A,
                B=rB,
                l2_B=l2B,
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

#vid -> massの辞書に変換する
def measure_to_dict(m: List[Tuple[int, float]]) -> Dict[int, float]:
    acc: Dict[int, float] = {}
    for vid, w in m:
        vid = int(vid)
        acc[vid] = acc.get(vid, 0.0) + float(w)
    return acc

#query と doc の measure を受け取って
def build_delta_masses(
    q_measure: List[Tuple[int, float]],
    d_measure: List[Tuple[int, float]],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
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

#greedy に prematch を行う関数。compare_rows は build_compare_rows の結果、plus_mass_in と minus_mass_in は build_delta_masses の結果。compare_rows を優先順位順に見ていき、u と picked v が両方とも残量があって、かつ pick された木の方が L2 で近ければマッチさせる。マッチさせたらその分だけ残量を減らす。マッチさせられなかった理由もカウントする。
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

#プレマッチのコスト
def prematch_cost_from_edges(match_edges: List[MatchEdge]) -> float:
    return float(sum(e.l2_pick * e.flow for e in match_edges))


# ============================================================
# 10) residual FlowTree cost
# ============================================================

def dict_to_measure(d: Dict[int, float], eps: float = 1e-15) -> List[Tuple[int, float]]:
    out = [(int(k), float(v)) for k, v in d.items() if float(v) > eps]
    out.sort(key=lambda x: x[0])
    return out


def sum_measure(m: List[Tuple[int, float]]) -> float:
    return float(sum(w for _, w in m))


def _balance_measures(
    a: List[Tuple[int, float]],
    b: List[Tuple[int, float]],
    balance_eps: float = 1e-9,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
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

#残った plus_mass と minus_mass を C++ 側の flowtree_distance_pair に渡してコストを計算する関数。balance_eps は、両方の measure の合計が十分近ければそのまま渡すための閾値。
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
# 11) 表示
# ============================================================

def print_summary(
    resA: SingleTreeResult,
    resB: SingleTreeResult,
    compare_summary: Dict[str, Any],
    doc_id: int,
    qid: int,
) -> None:
    print("[summary]")
    print(f"  qid={qid} doc_id={doc_id}")
    print(f"  treeA: seed={resA.seed} packedN={resA.packed.n_nodes} H={resA.height}")
    print(f"  treeB: seed={resB.seed} packedN={resB.packed.n_nodes} H={resB.height}")
    print(f"  priority = {compare_summary['priority']} (sort by d1 on treeA)")
    print(f"  common_u = {compare_summary['common_u']}")
    print(f"  onlyA_u  = {compare_summary['onlyA_u']}")
    print(f"  onlyB_u  = {compare_summary['onlyB_u']}")
    print(f"  same_NN(v_id_A==v_id_B) = {compare_summary['same_nn']}")
    print(f"  diff_NN(v_id_A!=v_id_B) = {compare_summary['diff_nn']}")
    print(f"  picked A = {compare_summary['picked_A']}")
    print(f"  picked B = {compare_summary['picked_B']}")
    print(f"  avg |L2_A-L2_B| = {compare_summary['avg_abs_l2_gap']:.6f}")
    print()


def print_compare_table(rows: List[CompareRow], limit: int = 200) -> None:
    print("[table] u-keyed compare & pick (priority by treeA d1 asc)")
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


def print_greedy_rows(match_edges: List[MatchEdge], show_top: int = 200) -> None:
    print(f"[matched] greedy flows (show_top={show_top})")
    print(" ord   u_vid   pick      v_vid   picked_d1      L2        flow       rem_u       rem_v")
    print("-" * 92)
    for e in match_edges[:show_top]:
        print(
            f"{e.ord_priority:4d} {e.u_id:7d} {e.pick:>6s} {e.v_id:10d} "
            f"{e.d1_pick:10d} {e.l2_pick:8.6f} {e.flow:10.6f} "
            f"{e.rem_u:10.6f} {e.rem_v:10.6f}"
        )
    print()


def print_flow_stats(delta_total: float, prematched: float, residual: float, nz_plus: int, nz_minus: int) -> None:
    ratio = 0.0 if delta_total <= 0 else prematched / delta_total
    print("[prematch flow]")
    print(f"  delta_total (sum plus/minus) = {delta_total:.6f}")
    print(f"  prematched_flow_sent         = {prematched:.6f}   (ratio={ratio:.3f})")
    print(f"  residual_mass                = {residual:.6f}   (#nonzero plus={nz_plus}, minus={nz_minus})")
    print()


def print_cost_block(seed_label: str, seed_value: int, baseline: float, prem_cost: float, residual: float) -> None:
    total = prem_cost + residual
    improvement = baseline - total
    print(f"[cost {seed_label}] (seed={seed_value})")
    print(f"  baseline_flowtree_cost      = {baseline:.6f}")
    print(f"  prematch_cost(L2*flow)      = {prem_cost:.6f}")
    print(f"  residual_flowtree_cost      = {residual:.6f}")
    print(f"  total_cost_with_prematch    = {total:.6f}")
    print(f"  improvement (baseline-total)= {improvement:.6f}")
    print()


# ============================================================
# 12) main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing vocab.npy / dataset.npy / queries.npy")
    ap.add_argument("--so_dir", required=True, help="directory containing ot_estimators_twotree*.so")
    ap.add_argument("--qid", type=int, default=0, help="query index into queries.npy")
    ap.add_argument("--doc_id", type=int, required=True, help="document id to inspect")
    ap.add_argument("--seedA", type=int, required=True, help="seed for tree A")
    ap.add_argument("--seedB", type=int, required=True, help="seed for tree B")
    ap.add_argument("--tie", choices=["left", "right"], default="left", help="tie-break for Algorithm B")
    ap.add_argument("--show_compare", action="store_true", help="print compare table")
    ap.add_argument("--show_match", action="store_true", help="print greedy prematch rows")
    ap.add_argument("--check_nn", action="store_true", help="check each tree's Algorithm B result against brute force")
    ap.add_argument("--limit_rows", type=int, default=200, help="row limit for printed tables")
    args = ap.parse_args()

    #データのロード
    vocab, dataset, queries = load_all_inputs(args.data_dir)

    #query と doc　を選ぶ
    if args.qid < 0 or args.qid >= len(queries):
        raise RuntimeError(f"qid out of range: {args.qid} (queries={len(queries)})")
    if args.doc_id < 0 or args.doc_id >= len(dataset):
        raise RuntimeError(f"doc_id out of range: {args.doc_id} (dataset={len(dataset)})")

    query = queries[args.qid]
    doc = dataset[args.doc_id]

    #二本の木を作る
    otA, otB, seed_api_ok = build_two_ots(
        so_dir=args.so_dir,
        vocab=vocab,
        seedA=args.seedA,
        seedB=args.seedB,
    )

    #木Aと木Bの処理
    resA = run_single_tree(
        ot=otA,
        query=query,
        doc=doc,
        seed=args.seedA,
        tie_break=args.tie,
    )
    resB = run_single_tree(
        ot=otB,
        query=query,
        doc=doc,
        seed=args.seedB,
        tie_break=args.tie,
    )

    #A/Bの候補を比較
    rows, compare_summary = build_compare_rows(resA.nn_scan, resB.nn_scan, vocab, priority="A")
    print_summary(resA, resB, compare_summary, args.doc_id, args.qid)

    if args.show_compare:
        print_compare_table(rows, limit=args.limit_rows)

    #元のquery/doc の delta mass を作る
    plus0, minus0, _ = build_delta_masses(query, doc)
    delta_total = float(sum(plus0.values()))

    #プレマッチとそのコスト計算
    match_edges, rem_plus, rem_minus, greedy_stats = greedy_prematch_delta_mass(rows, plus0, minus0)
    prem_cost = prematch_cost_from_edges(match_edges)
    residual_mass = float(greedy_stats["sum_rem_plus"])
    prematched_flow = float(greedy_stats["total_flow_sent"])

    if args.show_match:
        print_greedy_rows(match_edges, show_top=args.limit_rows)

    print_flow_stats(
        delta_total=delta_total,
        prematched=prematched_flow,
        residual=residual_mass,
        nz_plus=int(greedy_stats["nonzero_rem_plus"]),
        nz_minus=int(greedy_stats["nonzero_rem_minus"]),
    )

    print()
    print("DEBUG file =", __file__)
    print("DEBUG qid/doc_id =", args.qid, args.doc_id)
    print("DEBUG seedA/seedB =", args.seedA, args.seedB)
    print("DEBUG baselineA_direct =", float(otA.flowtree_distance_pair(query, doc)))
    print("DEBUG baselineB_direct =", float(otB.flowtree_distance_pair(query, doc)))
    print("DEBUG prem_cost =", prem_cost)
    print("DEBUG rem_plus_sum =", sum(rem_plus.values()))
    print("DEBUG rem_minus_sum =", sum(rem_minus.values()))

    print("DEBUG A1 =", float(otA.flowtree_distance_pair(query, doc)))
    print("DEBUG A2 =", float(otA.flowtree_distance_pair(query, doc)))
    print("DEBUG A3 =", float(otA.flowtree_distance_pair(query, doc)))

    print("DEBUG B1 =", float(otB.flowtree_distance_pair(query, doc)))
    print("DEBUG B2 =", float(otB.flowtree_distance_pair(query, doc)))
    print("DEBUG B3 =", float(otB.flowtree_distance_pair(query, doc)))
    print()

    baselineA = float(otA.flowtree_distance_pair(query, doc))
    residualA = flowtree_cost_pair(otA, rem_plus, rem_minus)
    print_cost_block("A", args.seedA, baselineA, prem_cost, residualA)
    baselineB = float(otB.flowtree_distance_pair(query, doc))
    residualB = flowtree_cost_pair(otB, rem_plus, rem_minus)
    print_cost_block("B", args.seedB, baselineB, prem_cost, residualB)

    if not seed_api_ok:
        print("[note] C++ 側の load_vocabulary は seed 引数を受け取っていません。")
        print("[note] 今回は seedA == seedB 前提で同一木として実行されています。")
        print()


if __name__ == "__main__":
    main()