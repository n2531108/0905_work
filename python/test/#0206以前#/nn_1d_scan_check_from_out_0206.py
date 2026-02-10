#!/usr/bin/env python3
# nn_1d_scan_check_from_out_0206.py
#
# 目的：
#   embed1d_algoA_demo.py が出力した *.out（例: adj_001.out）を読み、
#   packed木に含まれる 10語彙を
#     - クエリ側 5語彙 (query.npz の ids)
#     - 文書側 5語彙 (10語彙 - クエリ語彙)
#   に自動分解して、
#
#   「1D上で、X（クエリ）各点の最近傍をY（文書）から 2ポインタ1パス走査で求める」
#   が本当に正しい（＝全探索と一致する）かを検証する。
#
# 使い方例：
#   python3 nn_1d_scan_check_from_out_0206.py \
#     --data_dir "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo" \
#     --out "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo/emb_A/adj_001.out" \
#     --tie left --print
#
#   B側でも同様に：
#   python3 nn_1d_scan_check_from_out_0206.py --data_dir ... --out .../emb_B/adj_001.out --check
#
# オプション：
#   --check   : scan と brute を比較して一致するか確認（最重要）
#   --print   : 結果を表示
#   --bothdir : X->Y だけでなく Y->X も確認（どちらも最近傍が正しいか）
#
# 注意：
#   ここで検証しているのは「最近傍探索（NN）」であり、
#   OTの“流量を流すscan”や“AlgoB(OT)”とは別物です。

from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np


# -----------------------------
# embed1d_algoA_demo.py の .out パース
# -----------------------------
_LEAF_RE = re.compile(
    r'^\s*(?P<name>[A-Za-z0-9_]+)\s+\(id=\s*(?P<id>\d+)\)\s+x=\s*(?P<x>-?\d+)\s+depth=\s*(?P<depth>-?\d+)\s*$'
)

def parse_embed_out(path: str) -> Dict[int, int]:
    """
    .out から v{vid} の葉だけ拾って (vid -> x) を返す。
    例: ' v47  (id= 0)  x=  0  depth=2' から 47->0 を得る。
    """
    f: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            m = _LEAF_RE.match(line.rstrip("\n"))
            if not m:
                continue
            name = m.group("name")
            x = int(m.group("x"))
            if len(name) >= 2 and name[0] == "v" and name[1:].isdigit():
                vid = int(name[1:])
                f[vid] = x
    if not f:
        raise RuntimeError(f"[parse] no v* leaves found in: {path}")
    return f


# -----------------------------
# dataset.npz / query.npz 読み取り
# -----------------------------
def load_query_npz(path: str) -> Tuple[List[int], List[float]]:
    z = np.load(path)
    ids = z["ids"].astype(np.int64).tolist()
    ws  = z["weights"].astype(np.float64).tolist()
    return ids, ws

def load_dataset_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path)
    ids = z["ids"].astype(np.int64)
    ws  = z["weights"].astype(np.float64)
    offs = z["offsets"].astype(np.int64)
    return ids, ws, offs


# -----------------------------
# 1D NN：2ポインタ走査（Algorithm Bの“最近傍探索”部分）
# -----------------------------
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
    d1: int  # |u_x - v_x|

def sort_pts(pts: Iterable[Pt]) -> List[Pt]:
    return sorted(list(pts), key=lambda p: (p.x, p.id))

def abs_i(a: int) -> int:
    return a if a >= 0 else -a

def _pick_between_two(u: Pt, left: Optional[Pt], right: Optional[Pt], tie_break: str) -> NNResult:
    assert left is not None or right is not None
    if left is None:
        d = abs_i(u.x - right.x)
        return NNResult(u.id, u.x, right.id, right.x, d)
    if right is None:
        d = abs_i(u.x - left.x)
        return NNResult(u.id, u.x, left.id, left.x, d)

    dl = abs_i(u.x - left.x)
    dr = abs_i(u.x - right.x)

    if dl < dr:
        return NNResult(u.id, u.x, left.id, left.x, dl)
    if dr < dl:
        return NNResult(u.id, u.x, right.id, right.x, dr)

    # 同距離なら tie_break に従う
    if tie_break == "right":
        return NNResult(u.id, u.x, right.id, right.x, dr)
    return NNResult(u.id, u.x, left.id, left.x, dl)

def nn_scan_algoB(X: List[Pt], Y: List[Pt], tie_break: str = "left") -> List[NNResult]:
    """
    Xの各点に対して、Yの最近傍（|x-y|最小）を 2ポインタ1パス走査で求める。
    ※ X,Y は内部でソートする（未ソートでもOK）。
    """
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
        while j + 1 < m and Ys[j + 1].x <= u.x:
            j += 1

        if Ys[j].x <= u.x:
            left_cand = Ys[j]
            right_cand = Ys[j + 1] if (j + 1 < m) else None
        else:
            left_cand = None
            right_cand = Ys[0]

        res.append(_pick_between_two(u, left_cand, right_cand, tie_break=tie_break))

    return res

def nn_bruteforce(X: List[Pt], Y: List[Pt], tie_break: str = "left") -> List[NNResult]:
    """
    全探索（O(|X||Y|)）。scan版の正しさ検証用。
    tie-break は scan と揃える。
    """
    if not Y:
        raise ValueError("Y が空なので最近傍が定義できません")
    if tie_break not in ("left", "right"):
        raise ValueError("tie_break は 'left' か 'right' を指定してください")

    Xs = sort_pts(X)
    Ys = sort_pts(Y)

    out: List[NNResult] = []
    for u in Xs:
        best: Optional[Pt] = None
        best_key = None

        for v in Ys:
            d = abs_i(u.x - v.x)
            if tie_break == "left":
                key = (d, v.x, v.id)     # 同距離なら左（小さいx）優先
            else:
                key = (d, -v.x, -v.id)   # 同距離なら右（大きいx）優先
            if best_key is None or key < best_key:
                best_key = key
                best = v

        out.append(NNResult(u.id, u.x, best.id, best.x, abs_i(u.x - best.x)))
    return out

def check_equal(scan: List[NNResult], brute: List[NNResult]) -> Tuple[bool, str]:
    if len(scan) != len(brute):
        return False, f"len mismatch: scan={len(scan)} brute={len(brute)}"
    sm = {r.u_id: r for r in scan}
    bm = {r.u_id: r for r in brute}
    if set(sm.keys()) != set(bm.keys()):
        return False, "u_id set mismatch"
    for u in sorted(sm.keys()):
        a, b = sm[u], bm[u]
        if (a.v_id, a.d1, a.v_x) != (b.v_id, b.d1, b.v_x):
            return False, f"u={u}: scan(v={a.v_id},d={a.d1},vx={a.v_x}) != brute(v={b.v_id},d={b.d1},vx={b.v_x})"
    return True, "OK"


# -----------------------------
# doc 側 5語彙を dataset.npz から“特定”する（任意）
# -----------------------------
def build_doc_map(dataset_ids: np.ndarray, offsets: np.ndarray) -> Dict[Tuple[int, ...], List[int]]:
    """
    各 doc の語彙IDs(ソート済みタプル) -> doc_id のリスト を作る。
    num_docs=5000, words_per_doc=5 程度なら一瞬。
    """
    mp: Dict[Tuple[int, ...], List[int]] = {}
    n = len(offsets) - 1
    for doc_id in range(n):
        s, e = int(offsets[doc_id]), int(offsets[doc_id + 1])
        ids = tuple(sorted(dataset_ids[s:e].tolist()))
        mp.setdefault(ids, []).append(doc_id)
    return mp


def print_nn_list(title: str, rr: List[NNResult]):
    print(title)
    print(" u_id  u_x   ->   v_id  v_x   d1")
    for r in sorted(rr, key=lambda t: (t.u_x, t.u_id)):
        print(f"{r.u_id:>4d} {r.u_x:>4d}        {r.v_id:>4d} {r.v_x:>4d} {r.d1:>4d}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="tmp_flowtree_demo dir (vocab.npy, dataset.npz, query.npz)")
    ap.add_argument("--out", required=True, help="embed output (.out), e.g. emb_A/adj_001.out")
    ap.add_argument("--tie", choices=["left", "right"], default="left", help="同距離時の優先（デフォルト left）")
    ap.add_argument("--check", action="store_true", help="scan と brute が一致するか確認")
    ap.add_argument("--print", dest="do_print", action="store_true", help="結果を表示")
    ap.add_argument("--bothdir", action="store_true", help="X->Y だけでなく Y->X も検証する")
    ap.add_argument("--find_doc", action="store_true", help="文書側5語彙が dataset の何番docか特定して表示する")
    args = ap.parse_args()

    # 1) .out から 10語彙（vid->x）を取得
    f = parse_embed_out(args.out)
    S = sorted(f.keys())
    print(f"[out parsed] #leaves={len(S)}  S={S}")

    # 2) query.npz からクエリ側 5語彙 を取得
    q_ids, q_ws = load_query_npz(os.path.join(args.data_dir, "query.npz"))
    q_set = set(q_ids)
    print(f"[query] ids={q_ids}")
    print(f"[query] sum_w={float(np.sum(q_ws)):.6f}")

    # 3) S の中でクエリ側/文書側 に分解
    X_ids = [vid for vid in S if vid in q_set]
    Y_ids = [vid for vid in S if vid not in q_set]

    if len(X_ids) == 0 or len(Y_ids) == 0:
        raise RuntimeError(
            f"split failed: X_ids={X_ids} Y_ids={Y_ids}. "
            f"（out内の葉と query ids が合っているか確認して）"
        )

    print(f"[split] X(query side) ids={X_ids}  (#={len(X_ids)})")
    print(f"[split] Y(doc side)   ids={Y_ids}  (#={len(Y_ids)})")

    # 4) X, Y を Pt 化（x座標は .out から）
    X = [Pt(vid, f[vid]) for vid in X_ids]
    Y = [Pt(vid, f[vid]) for vid in Y_ids]

    # 5) 最近傍（scan）と（brute）を計算
    scan_xy = nn_scan_algoB(X, Y, tie_break=args.tie)
    brute_xy = nn_bruteforce(X, Y, tie_break=args.tie)

    if args.do_print:
        print_nn_list("=== X(query) -> Y(doc)  scan NN ===", scan_xy)

    if args.check:
        ok, msg = check_equal(scan_xy, brute_xy)
        print(f"[check X->Y] {msg}")
        if not ok:
            print_nn_list("scan (X->Y)", scan_xy)
            print_nn_list("brute(X->Y)", brute_xy)
            raise SystemExit(1)

    if args.bothdir:
        scan_yx = nn_scan_algoB(Y, X, tie_break=args.tie)
        brute_yx = nn_bruteforce(Y, X, tie_break=args.tie)

        if args.do_print:
            print_nn_list("=== Y(doc) -> X(query)  scan NN ===", scan_yx)

        if args.check:
            ok, msg = check_equal(scan_yx, brute_yx)
            print(f"[check Y->X] {msg}")
            if not ok:
                print_nn_list("scan (Y->X)", scan_yx)
                print_nn_list("brute(Y->X)", brute_yx)
                raise SystemExit(1)

    # 6) （任意）dataset からこの文書が何番か当てる
    if args.find_doc:
        ds_ids, ds_w, offs = load_dataset_npz(os.path.join(args.data_dir, "dataset.npz"))
        mp = build_doc_map(ds_ids, offs)

        key = tuple(sorted(Y_ids))
        doc_ids = mp.get(key, [])
        if not doc_ids:
            print(f"[find_doc] NOT FOUND for doc-vocab={list(key)}")
        else:
            print(f"[find_doc] found doc_ids={doc_ids} for doc-vocab={list(key)} (show first)")
            doc0 = doc_ids[0]
            s, e = int(offs[doc0]), int(offs[doc0 + 1])
            pairs = list(zip(ds_ids[s:e].tolist(), ds_w[s:e].tolist()))
            pairs = [(int(i), float(w)) for (i, w) in pairs]
            print(f"[find_doc] doc[{doc0}] pairs={pairs}  sum_w={sum(w for _, w in pairs):.6f}")

    print("[done]")


if __name__ == "__main__":
    main()
