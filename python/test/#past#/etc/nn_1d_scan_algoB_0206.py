#!/usr/bin/env python3
# nn_1d_scan_algoB_0206.py
#
# Algorithm B（付録のアルゴリズムB想定）：
#   1次元に埋め込まれた点列について、ソート済み座標を「1パス走査（2ポインタ）」で
#   最近傍を高速に見つける手法です。
#
# 典型的な状況：
#   - X（クエリ側の葉）と Y（相手側の葉）がそれぞれ 1D 座標を持っている
#   - X, Y を座標順にソート済みとみなせる
#   - 各 x ∈ X に対して、最も近い y ∈ Y（|x-y| 最小）を求めたい
#
# 計算量：
#   - ソート済みの前提なら O(|X| + |Y|)
#   - ソートが必要なら O(|X|log|X| + |Y|log|Y|) + 走査 O(|X|+|Y|)
#
# 付録：
#   Flowtree 的には「符号が異なる葉同士」しか流せないので、
#   +側からは -側の最近傍、-側からは +側の最近傍、という形の“符号付き最近傍”も用意します。
#
# 使い方例：
#   python3 nn_1d_scan_algoB_0206.py --demo
#   python3 nn_1d_scan_algoB_0206.py --random 50 80 --seed 0 --check
#   python3 nn_1d_scan_algoB_0206.py --signed-demo
#
# 注意：
#   距離が同点（左右どちらも同じ距離）になる場合の決め方（tie-break）を選べます。
#   デフォルトは "left"（座標が小さい方＝左側を優先）。

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable


# -----------------------------
# データ構造
# -----------------------------
@dataclass(frozen=True)
class Pt:
    """1次元点。id は葉IDなど、x が1D座標。"""
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
# ユーティリティ
# -----------------------------
def sort_pts(pts: Iterable[Pt]) -> List[Pt]:
    """座標 x で昇順、同座標なら id 昇順で安定ソート。"""
    return sorted(list(pts), key=lambda p: (p.x, p.id))


def abs_i(a: int) -> int:
    return a if a >= 0 else -a


def _pick_between_two(
    u: Pt,
    left: Optional[Pt],
    right: Optional[Pt],
    tie_break: str = "left",
) -> NNResult:
    """
    left / right のどちらか（または片方が None）から、u に近い方を選ぶ。
    同点なら tie_break に従う。
    """
    assert left is not None or right is not None, "候補が両方 None はありえません"

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

    # 同点（u の左右が等距離）
    if tie_break == "right":
        return NNResult(u.id, u.x, right.id, right.x, dr)
    return NNResult(u.id, u.x, left.id, left.x, dl)


# -----------------------------
# Algorithm B：1パス走査で最近傍
# -----------------------------
def nn_scan_algoB(
    X: List[Pt],
    Y: List[Pt],
    tie_break: str = "left",
) -> List[NNResult]:
    """
    X の各点に対し、Y の最近傍（|x-y| 最小）を 1パス走査で求める。

    考え方：
      - Y はソート済みとする（ここで内部的にソートしてOK）
      - X もソートして左から順に処理する
      - ポインタ j を「Y[j] が X[i] の左側にある最大の位置（床：floor）」として更新していく
      - X[i] の候補は基本的に「Y[j]（左側）と Y[j+1]（右側）」の2つだけ
        ※単調性により、それ以外は必ず距離が大きい

    返り値：
      X を座標順に処理した順の NNResult のリスト。
      （元の順番に戻したければ u_id で辞書化してください）
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
        # 次の点が u.x 以下である限り、j を右に進める
        # 結果として「Ys[j].x <= u.x < Ys[j+1].x」を満たす境界に落ちる（端を除く）
        while j + 1 < m and Ys[j + 1].x <= u.x:
            j += 1

        # u.x より左にある候補（floor）と、右にある候補（ceil）を作る
        if Ys[j].x <= u.x:
            left_cand = Ys[j]
            right_cand = Ys[j + 1] if (j + 1 < m) else None
        else:
            # u.x < Ys[0].x のケース（全部右側）
            left_cand = None
            right_cand = Ys[0]

        res.append(_pick_between_two(u, left_cand, right_cand, tie_break=tie_break))

    return res


# -----------------------------
# 正しさ確認用：全探索（遅いが確実）
# -----------------------------
def nn_bruteforce(
    X: List[Pt],
    Y: List[Pt],
    tie_break: str = "left",
) -> List[NNResult]:
    """
    X の各点に対して Y 全点をなめて最近傍を探す（O(|X||Y|)）。
    scan 版と一致するかを確認するために使う。
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

        if tie_break == "left":
            # 同距離なら x が小さい方（左）を優先
            best_key = None
            for v in Ys:
                d = abs_i(u.x - v.x)
                key = (d, v.x, v.id)  # 安定
                if best_key is None or key < best_key:
                    best_key = key
                    best_pt = v
        else:
            # 同距離なら x が大きい方（右）を優先
            best_key = None
            for v in Ys:
                d = abs_i(u.x - v.x)
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
# Flowtree向け：符号で分割して「異符号最近傍」を取る
# -----------------------------
def split_by_sign(
    pts: List[Pt],
    sign_map: dict[int, int],
) -> Tuple[List[Pt], List[Pt]]:
    """
    sign_map: id -> +1 / -1（0 は無視）
    戻り値: (plus_pts, minus_pts)
    """
    plus, minus = [], []
    for p in pts:
        s = sign_map.get(p.id, 0)
        if s > 0:
            plus.append(p)
        elif s < 0:
            minus.append(p)
        else:
            # 0 は今回無視（必要なら別処理）
            pass
    return plus, minus


def nn_signed_opposite(
    pts: List[Pt],
    sign_map: dict[int, int],
    tie_break: str = "left",
) -> List[NNResult]:
    """
    異符号同士で最近傍を取る：
      - +側の各点に対して、-側の最近傍
      - -側の各点に対して、+側の最近傍
    を両方返す（連結したリスト）。
    """
    plus, minus = split_by_sign(pts, sign_map)

    out: List[NNResult] = []
    if plus and minus:
        out.extend(nn_scan_algoB(plus, minus, tie_break=tie_break))
        out.extend(nn_scan_algoB(minus, plus, tie_break=tie_break))
    return out


# -----------------------------
# 表示
# -----------------------------
def print_results(title: str, rr: List[NNResult], limit: Optional[int] = None):
    print(title)
    print(" u_id  u_x   ->   v_id  v_x   d1")
    for i, r in enumerate(rr):
        if limit is not None and i >= limit:
            print(f"... （残り {len(rr)-limit} 件）")
            break
        print(f"{r.u_id:>4d} {r.u_x:>4d}        {r.v_id:>4d} {r.v_x:>4d} {r.d1:>4d}")
    print()


# -----------------------------
# デモ / 乱数生成
# -----------------------------
def make_random_two_sets(n: int, m: int, lo: int, hi: int, seed: int) -> Tuple[List[Pt], List[Pt]]:
    rng = random.Random(seed)
    X = [Pt(i, rng.randint(lo, hi)) for i in range(n)]
    Y = [Pt(10_000 + j, rng.randint(lo, hi)) for j in range(m)]
    return X, Y


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="固定の小デモを走らせる")
    ap.add_argument("--signed-demo", action="store_true", help="符号付き（異符号最近傍）の小デモを走らせる")
    ap.add_argument("--random", nargs=2, type=int, metavar=("N", "M"),
                    help="乱数で X を N 個、Y を M 個生成して scan を実行")
    ap.add_argument("--range", dest="rng", nargs=2, type=int, default=[0, 50], metavar=("LO", "HI"),
                    help="--random 時の座標範囲（デフォルト 0..50）")
    ap.add_argument("--seed", type=int, default=0, help="乱数seed")
    ap.add_argument("--tie", choices=["left", "right"], default="left", help="同距離時の優先（デフォルト left）")
    ap.add_argument("--check", action="store_true", help="全探索（brute）と一致するか確認する")
    ap.add_argument("--print", dest="do_print", action="store_true", help="結果を表示する（random のときは任意）")
    args = ap.parse_args()

    if args.demo:
        # 小さな例：X と Y が混在する状況
        X = [Pt(1, 0), Pt(2, 5), Pt(3, 6), Pt(4, 10)]
        Y = [Pt(101, 2), Pt(102, 7), Pt(103, 11)]
        scan = nn_scan_algoB(X, Y, tie_break=args.tie)
        print_results("=== demo: scan 結果 ===", scan)
        if args.check:
            brute = nn_bruteforce(X, Y, tie_break=args.tie)
            ok, msg = check_equal(scan, brute)
            print("check:", msg)
        return

    if args.signed_demo:
        # Flowtree 的な「異符号同士」のイメージデモ
        pts = [
            Pt(10, 0), Pt(11, 4), Pt(12, 9), Pt(13, 12),
            Pt(20, 1), Pt(21, 5), Pt(22, 10),
        ]
        sign_map = {
            10: +1, 11: +1, 12: +1, 13: +1,
            20: -1, 21: -1, 22: -1,
        }
        rr = nn_signed_opposite(pts, sign_map, tie_break=args.tie)
        rr = sorted(rr, key=lambda r: (r.u_x, r.u_id))
        print_results("=== signed demo: 異符号最近傍（+→- と -→+） ===", rr)
        return

    if args.random:
        n, m = args.random
        lo, hi = args.rng
        X, Y = make_random_two_sets(n, m, lo, hi, seed=args.seed)

        scan = nn_scan_algoB(X, Y, tie_break=args.tie)
        if args.do_print:
            print_results("=== random: scan 結果 ===", scan, limit=50)

        if args.check:
            brute = nn_bruteforce(X, Y, tie_break=args.tie)
            ok, msg = check_equal(scan, brute)
            if not ok:
                print("check: FAIL")
                print(msg)
                # 簡易デバッグ用に先頭だけ表示
                print("X (sorted head):", [(p.id, p.x) for p in sort_pts(X)[:20]])
                print("Y (sorted head):", [(p.id, p.x) for p in sort_pts(Y)[:20]])
                return
            print("check: OK")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
