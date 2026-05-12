#!/usr/bin/env python3
# nn_1d_scan_check_from_out_0210.py
#
# 目的：
#   embed1d_algoA_demo.py の出力 (*.out) から 1D座標 f(v)=x を読み取り、
#   クエリ側5語彙(Q) と 文書側5語彙(D) に分割して、
#   論文 Algorithm B（while条件式が完全に同じ版）で
#   「異集合間の最近傍（|x-y|最小）」が 1パス(2ポインタ)で求まることを確認する。
#
# 重要：
#   - Flowtree の packed 木には「クエリ側の葉」と「文書側の葉」が混在する
#   - 今回は query.npz の ids を用いて「クエリ側 / 文書側」を分ける（方式A）
#   - 最近傍は 1D距離 |x_u - x_v| で定義（同距離の tie-break は left/right 選択可）
#   - --check で brute force と一致するか検証
#
# 使い方例：
#   # 片方向（X=query -> Y=doc）のみ、表示しつつチェック
#   python3 nn_1d_scan_check_from_out_0210.py \
#     --data_dir "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo" \
#     --out "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo/emb_A/adj_001.out" \
#     --print --check
#
#   # 両方向（X->Y と Y->X）を表示しつつチェック
#   python3 nn_1d_scan_check_from_out_0210.py \
#     --data_dir "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo" \
#     --out "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo/emb_A/adj_001.out" \
#     --print --check --bothdir
#
#   # B側(out)でも同様
#   python3 nn_1d_scan_check_from_out_0210.py \
#     --data_dir "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo" \
#     --out "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo/emb_B/adj_001.out" \
#     --print --check --bothdir
#
#   # tie-break を右優先に（同距離なら右側を選ぶ）
#   python3 nn_1d_scan_check_from_out_0210.py ... --tie right --print --check
#
# 出力の見方：
#   - [split] で X(query側) と Y(doc側) の id が表示される
#   - 表では u_id(u_x) -> v_id(v_x) d1=|u_x-v_x|
#   - [check ...] OK なら 1パスscan結果が brute と一致（＝正しく最近傍が取れている）
#
from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np


# -----------------------------
# データ構造
# -----------------------------
@dataclass(frozen=True)
class Pt:
    """1次元点。id は語彙ID、x は 1D座標（埋め込み結果）。"""
    id: int
    x: int


@dataclass(frozen=True)
class NNResult:
    """最近傍の結果（u -> v）。"""
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int  # |u_x - v_x|


# -----------------------------
# parsing: embed1d_algoA_demo.py output (*.out)
# -----------------------------
# 例：
#   v47  (id= 0)  x=  0  depth=2
_LEAF_RE = re.compile(
    r'^\s*(?P<name>[A-Za-z0-9_]+)\s+\(id=\s*(?P<id>\d+)\)\s+x=\s*(?P<x>-?\d+)\s+depth=\s*(?P<depth>-?\d+)\s*$'
)


def parse_embed_out(path: str) -> Tuple[dict[int, int], dict[int, Tuple[int, int, str]]]:
    """
    Returns:
      f: dict[vocab_id -> x]
      meta: dict[vocab_id -> (node_id, depth, name)]
    NOTE:
      - name が v{int} の行だけを採用する（例: v47）
      - f は「語彙ID -> 1D座標」を返す
    """
    f: dict[int, int] = {}
    meta: dict[int, Tuple[int, int, str]] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            m = _LEAF_RE.match(line.rstrip("\n"))
            if not m:
                continue

            name = m.group("name")
            node_id = int(m.group("id"))
            x = int(m.group("x"))
            depth = int(m.group("depth"))

            if len(name) >= 2 and name[0] == "v" and name[1:].isdigit():
                vid = int(name[1:])
                f[vid] = x
                meta[vid] = (node_id, depth, name)

    if not f:
        raise RuntimeError(f"[parse] no v* leaves found in: {path}")
    return f, meta


def load_query_ids(data_dir: str) -> List[int]:
    """data_dir/query.npz からクエリ語彙IDを読む。"""
    qpath = os.path.join(data_dir, "query.npz")
    if not os.path.exists(qpath):
        raise RuntimeError(f"missing query.npz: {qpath}")
    npz = np.load(qpath)
    if "ids" not in npz:
        raise RuntimeError(f"query.npz missing 'ids': {qpath}")
    ids = [int(x) for x in npz["ids"].tolist()]
    return ids


def load_query_weights(data_dir: str) -> List[float]:
    """デバッグ用：queryのweightsも読む（sum確認など）。"""
    qpath = os.path.join(data_dir, "query.npz")
    npz = np.load(qpath)
    if "weights" not in npz:
        return []
    ws = [float(x) for x in npz["weights"].tolist()]
    return ws


# -----------------------------
# 1D 最近傍：論文 Algorithm B（while条件式を同じにする）
# -----------------------------
def sort_pts(pts: Iterable[Pt]) -> List[Pt]:
    """座標 x 昇順、同座標なら id 昇順の安定ソート。"""
    return sorted(list(pts), key=lambda p: (p.x, p.id))


def abs_i(a: int) -> int:
    return a if a >= 0 else -a


def nn_scan_algoB_pseudocode(
    X: List[Pt],
    Y: List[Pt],
    tie_break: str = "left",
) -> List[NNResult]:
    """
    論文の疑似コードと「同じ while 条件式」で書いた Algorithm B。

    原文（概念）：
      j = 1
      for i:
        while j < m and |f(y_{j+1})-f(x_i)| < |f(y_j)-f(x_i)|:
            j = j + 1
        τ(i) = j

    実装上：
      - 0-indexなので j=0 から開始
      - tie_break="left" は論文の < と整合（同距離なら進まない）
      - tie_break="right" を使いたい場合だけ、同距離を右に寄せる拡張を後段で入れる
        （※これは論文のままではないが、オプションとして便利）
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
        # 論文と同じ条件式：
        # while j < m and |y_{j+1}-x_i| < |y_j-x_i|:
        while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) < abs_i(Ys[j].x - u.x)):
            j += 1

        # τ(i)=j：最近傍は Ys[j]
        v = Ys[j]
        d = abs_i(v.x - u.x)

        # right tie-break 拡張：同距離なら右側へ寄せる
        if tie_break == "right":
            while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) == abs_i(Ys[j].x - u.x)):
                j += 1
                v = Ys[j]
                d = abs_i(v.x - u.x)

        res.append(NNResult(u.id, u.x, v.id, v.x, d))

    return res


# -----------------------------
# 正しさ確認用：全探索（brute）
# -----------------------------
def nn_bruteforce(
    X: List[Pt],
    Y: List[Pt],
    tie_break: str = "left",
) -> List[NNResult]:
    """
    X の各点に対して Y 全点をなめて最近傍を探す（O(|X||Y|)）。
    scan 版と一致するか確認するために使う。

    tie-break の仕様：
      - left: 同距離なら x が小さい方（左）、さらに id が小さい方
      - right: 同距離なら x が大きい方（右）、さらに id が大きい方
    """
    if not Y:
        raise ValueError("Y が空なので最近傍が定義できません")
    if tie_break not in ("left", "right"):
        raise ValueError("tie_break は 'left' か 'right' を指定してください")

    Xs = sort_pts(X)
    Ys = sort_pts(Y)

    out: List[NNResult] = []
    for u in Xs:
        best_pt: Optional[Pt] = None
        best_key = None

        for v in Ys:
            d = abs_i(u.x - v.x)
            if tie_break == "left":
                key = (d, v.x, v.id)
            else:
                key = (d, -v.x, -v.id)
            if best_key is None or key < best_key:
                best_key = key
                best_pt = v

        out.append(NNResult(u.id, u.x, best_pt.id, best_pt.x, abs_i(u.x - best_pt.x)))
    return out


def check_equal(scan: List[NNResult], brute: List[NNResult]) -> Tuple[bool, str]:
    """scan と brute が u_id ごとに一致するか検証する。"""
    if len(scan) != len(brute):
        return False, f"長さが違います: scan={len(scan)} brute={len(brute)}"

    sm = {r.u_id: r for r in scan}
    bm = {r.u_id: r for r in brute}
    if set(sm.keys()) != set(bm.keys()):
        return False, "u_id の集合が一致しません"

    for u in sorted(sm.keys()):
        a = sm[u]
        b = bm[u]
        if (a.v_id, a.d1, a.v_x) != (b.v_id, b.d1, b.v_x):
            return (
                False,
                f"u={u} で不一致: scan->(v={a.v_id},d={a.d1},vx={a.v_x}) "
                f"brute->(v={b.v_id},d={b.d1},vx={b.v_x})",
            )
    return True, "OK"


# -----------------------------
# split: query側 / doc側（方式A）
# -----------------------------
def split_query_doc(
    S: List[int],
    query_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    out に出てきた語彙集合 S を、
      - X = query側（query_ids に含まれるもの）
      - Y = doc側（それ以外）
    に分割する（方式A）。
    """
    qset = set(query_ids)
    X = [v for v in S if v in qset]
    Y = [v for v in S if v not in qset]
    return X, Y


# -----------------------------
# 表示
# -----------------------------
def print_results(title: str, rr: List[NNResult]):
    print(title)
    print(" u_id  u_x   ->   v_id  v_x   d1")
    for r in rr:
        print(f"{r.u_id:>4d} {r.u_x:>4d}        {r.v_id:>4d} {r.v_x:>4d} {r.d1:>4d}")
    print()


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir containing query.npz (and optionally vocab.npy)")
    ap.add_argument("--out", required=True, help="embed1d_algoA_demo.py output (.out) path")
    ap.add_argument("--tie", choices=["left", "right"], default="left",
                    help="同距離時の優先。論文通りは left（< で進めない）")
    ap.add_argument("--check", action="store_true", help="scan と brute が一致するか確認する")
    ap.add_argument("--print", dest="do_print", action="store_true", help="結果を表示する")
    ap.add_argument("--bothdir", action="store_true",
                    help="両方向（X->Y と Y->X）を出す。指定なしだと X->Y のみ。")
    args = ap.parse_args()

    # out を読む（語彙ID -> x座標）
    f, _ = parse_embed_out(args.out)
    S = sorted(f.keys())

    print(f"[out parsed] #leaves={len(S)}  S={S}")

    # query を読む
    q_ids = load_query_ids(args.data_dir)
    q_ws = load_query_weights(args.data_dir)
    if q_ws:
        print(f"[query] ids={q_ids}")
        print(f"[query] sum_w={sum(q_ws):.6f}")
    else:
        print(f"[query] ids={q_ids}")

    # split (方式A)
    X_ids, Y_ids = split_query_doc(S, q_ids)
    print(f"[split] X(query side) ids={sorted(X_ids)}  (#={len(X_ids)})")
    print(f"[split] Y(doc side)   ids={sorted(Y_ids)}  (#={len(Y_ids)})")

    if len(X_ids) == 0 or len(Y_ids) == 0:
        raise RuntimeError("X または Y が空です。split が崩れている可能性があります。")

    # Pt 化
    X_pts = [Pt(v, f[v]) for v in X_ids]
    Y_pts = [Pt(v, f[v]) for v in Y_ids]

    # ---- X -> Y ----
    scan_xy = nn_scan_algoB_pseudocode(X_pts, Y_pts, tie_break=args.tie)
    if args.do_print:
        print_results("=== X(query) -> Y(doc)  scan NN ===", scan_xy)

    if args.check:
        brute_xy = nn_bruteforce(X_pts, Y_pts, tie_break=args.tie)
        ok, msg = check_equal(scan_xy, brute_xy)
        print("[check X->Y]", msg)
        if not ok:
            # 簡易デバッグ（ソート済みを確認しやすい）
            Xs = sort_pts(X_pts)
            Ys = sort_pts(Y_pts)
            print("X(sorted):", [(p.id, p.x) for p in Xs])
            print("Y(sorted):", [(p.id, p.x) for p in Ys])
            return

    # ---- Y -> X（必要なら）----
    if args.bothdir:
        scan_yx = nn_scan_algoB_pseudocode(Y_pts, X_pts, tie_break=args.tie)
        if args.do_print:
            print_results("=== Y(doc) -> X(query)  scan NN ===", scan_yx)

        if args.check:
            brute_yx = nn_bruteforce(Y_pts, X_pts, tie_break=args.tie)
            ok, msg = check_equal(scan_yx, brute_yx)
            print("[check Y->X]", msg)
            if not ok:
                Xs = sort_pts(X_pts)
                Ys = sort_pts(Y_pts)
                print("X(sorted):", [(p.id, p.x) for p in Xs])
                print("Y(sorted):", [(p.id, p.x) for p in Ys])
                return

    print("[done]")


if __name__ == "__main__":
    main()
