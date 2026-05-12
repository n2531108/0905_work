#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flowtree_real_1d_nn_pipeline.py

目的:
  実データ（otdata_glove50_full）で FlowTree の packed subtree dump を取り、
  Algorithm A（1D tree embedding）で葉に 1D 座標 f(v)=x を割り当て、
  dump に含まれる signed mass（+/-）で葉を二分し、
  Algorithm B（論文の while 条件式と同じ）で + -> - 最近傍を 1パスで求める。
  さらに brute force と一致するかチェックする。

入出力:
  入力:
    data_dir/vocab.npy      (float32 2D)
    data_dir/dataset.npy    (object array 推奨: docごとの [(vid,w),...] など)
    data_dir/queries.npy    (object array 推奨: queryごとの [(vid,w),...] など)
  出力:
    - packed subtree dump: --dump_out で指定した txt
    - 1D 埋め込み out:     --emb_out_dir で指定した dir に docごと *.out 保存
      （あなたの旧 embed1d_algoA_demo.py の out 形式に寄せつつ sign/mass も追記）

使い方例:
  python3 flowtree_real_1d_nn_pipeline.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --so_dir   "/mnt/c/Users/成見/0905_work/native/build" \
    --seed 0 --qid 0 --docs "0,1,2" \
    --dump_out "/mnt/c/Users/成見/0905_work/tmp_real_dump/adj_dump_real.txt" \
    --dump_limit 3 \
    --emb_out_dir "/mnt/c/Users/成見/0905_work/tmp_real_dump/emb_real" \
    --assume_docs_in_order \
    --print --check

注意:
  - dataset.npy / queries.npy の内部形式が環境により違う可能性があるので、
    loader は「(vid,weight) の列」だけでなく「vidだけの列」なども吸収するように頑張ってます。
  - +/− は「FlowTree の差分（delta）」に由来するものとして扱います。
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Any
from collections import deque

import numpy as np


# ============================================================
# 1) データ構造
# ============================================================

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


@dataclass
class DumpBlock:
    """packed subtree dump 1ブロック分"""
    idx: int
    doc_id: Optional[int]
    dump_seed: Optional[int]

    N: int
    root: int
    edges: List[Tuple[int, int]]
    is_leaf: List[int]
    unleaf: List[int]

    # optional: packed node ごとの signed mass（delta）
    delta: Optional[List[float]] = None


# ============================================================
# 2) 実データ（dataset.npy / queries.npy）ローダ
# ============================================================

def _to_measure(obj: Any) -> List[Tuple[int, float]]:
    """
    「1つの測度（doc or query）」を Python list[(vid, weight)] に変換する。
    想定入力:
      - [(vid, w), ...]
      - np.ndarray shape (L,2)
      - [vid, vid, ...]（重み無し） -> 一様重みで正規化
      - np.ndarray shape (L,) の int -> 一様重みで正規化
    """
    if obj is None:
        return []

    # numpy scalar を素直に扱う
    if isinstance(obj, (np.integer, int)):
        # 1つだけの id の可能性 → それ自体を 1点測度とみなす
        vid = int(obj)
        return [(vid, 1.0)]

    # ndarray
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # object配列なら list化して再帰
            return _to_measure(obj.tolist())
        if obj.ndim == 2 and obj.shape[1] == 2:
            out = [(int(a), float(b)) for a, b in obj.tolist()]
            return _normalize_measure(out)
        if obj.ndim == 1:
            ids = [int(x) for x in obj.tolist()]
            return _uniform_measure(ids)
        # それ以外は list化して挑戦
        return _to_measure(obj.tolist())

    # list/tuple
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []

        # すでに (vid,w) の列？
        # 先頭が長さ2のタプル/リストならそう判断
        first = obj[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            out = []
            for t in obj:
                if not (isinstance(t, (list, tuple)) and len(t) == 2):
                    # 混ざってたら fallback
                    break
                out.append((int(t[0]), float(t[1])))
            if len(out) == len(obj):
                return _normalize_measure(out)

        # ただの id の列っぽい
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

        # 最後の手段: 1個ずつ変換を試みる（ただし危険）
        # ここで壊れるなら元データ形式を再確認が必要。
        out = []
        for x in obj:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                out.append((int(x[0]), float(x[1])))
            elif isinstance(x, (np.integer, int)):
                out.append((int(x), 0.0))
            else:
                raise TypeError(f"cannot interpret measure element: {type(x)} {x}")
        # 0.0 重みが混ざるのはおかしいので、ここは正規化せず返す
        return _normalize_measure(out)

    raise TypeError(f"cannot interpret measure: type={type(obj)}")


def _uniform_measure(ids: List[int]) -> List[Tuple[int, float]]:
    if len(ids) == 0:
        return []
    w = 1.0 / float(len(ids))
    out = [(int(v), w) for v in ids]
    return out


def _normalize_measure(m: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """重みが sum=1 になるように正規化。負は想定しない。"""
    if not m:
        return []
    s = float(sum(w for _, w in m))
    if s <= 0:
        # ありえないが、落とすよりは一様にする
        ids = [vid for vid, _ in m]
        return _uniform_measure(ids)
    return [(int(vid), float(w) / s) for vid, w in m]


def load_object_measures(path: str) -> List[List[Tuple[int, float]]]:
    """
    dataset.npy / queries.npy を読む。
    推奨形式:
      np.save(..., np.array(list_of_measures, dtype=object))
    """
    if not os.path.exists(path):
        raise RuntimeError(f"missing: {path}")

    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        measures = []
        for x in arr.tolist():
            measures.append(_to_measure(x))
        return measures

    # objectじゃないケース（例: 2D int の行列など）にも対応（雑に行を measure とみなす）
    if isinstance(arr, np.ndarray) and arr.ndim >= 1:
        measures = []
        for i in range(arr.shape[0]):
            measures.append(_to_measure(arr[i]))
        return measures

    # scalar
    return [_to_measure(arr)]


def load_vocab(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise RuntimeError(f"missing vocab.npy: {path}")
    v = np.load(path, allow_pickle=False)
    if not isinstance(v, np.ndarray) or v.ndim != 2 or v.dtype != np.float32:
        raise RuntimeError(f"vocab.npy must be float32 2D, got {type(v)} ndim={getattr(v,'ndim',None)} dtype={getattr(v,'dtype',None)}")
    return v


# ============================================================
# 3) dump パーサ（複数ブロック）
# ============================================================

_BEGIN_RE = re.compile(r"^#BEGIN_DUMP\s+idx=(\d+)(?:\s+doc_id=(\d+))?(?:\s+seed=([-\d]+))?\s*$")
_END_RE   = re.compile(r"^#END_DUMP\s+idx=(\d+).*$")

def parse_multi_dump(path: str) -> List[DumpBlock]:
    if not os.path.exists(path):
        raise RuntimeError(f"dump file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # ブロック境界: #BEGIN_DUMP があればそれ基準、無ければ #PACKED_NODES 基準
    begin_pos = [i for i, l in enumerate(lines) if l.startswith("#BEGIN_DUMP")]
    if begin_pos:
        blocks_raw = []
        for s, e in zip(begin_pos, begin_pos[1:] + [len(lines)]):
            blocks_raw.append(lines[s:e])
    else:
        starts = [i for i, l in enumerate(lines) if l.startswith("#PACKED_NODES")]
        if not starts:
            raise RuntimeError(f"no dump blocks found in: {path}")
        blocks_raw = []
        for s, e in zip(starts, starts[1:] + [len(lines)]):
            blocks_raw.append(lines[s:e])

    out: List[DumpBlock] = []
    for blk in blocks_raw:
        db = parse_one_block(blk)
        out.append(db)
    return out


def _parse_float_list(tokens: List[str]) -> List[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except Exception:
            # "nan" とかはそのままfloatで行けるのでここに来ないはず
            out.append(float("nan"))
    return out


def parse_one_block(lines: List[str]) -> DumpBlock:
    idx: int = -1
    doc_id: Optional[int] = None
    dump_seed: Optional[int] = None

    N: Optional[int] = None
    root: Optional[int] = None
    edges: List[Tuple[int, int]] = []
    is_leaf: List[int] = []
    unleaf: List[int] = []
    delta: Optional[List[float]] = None

    stage = None

    for s in lines:
        s = s.strip()
        if not s:
            continue

        m = _BEGIN_RE.match(s)
        if m:
            idx = int(m.group(1))
            doc_id = int(m.group(2)) if m.group(2) is not None else None
            dump_seed = int(m.group(3)) if m.group(3) is not None else None
            continue

        if s.startswith("#PACKED_NODES"):
            N = int(s.split()[1])
            continue
        if s.startswith("#ROOT"):
            root = int(s.split()[1])
            continue
        if s.startswith("#EDGE"):
            stage = "EDGE"
            continue
        if s.startswith("#ISLEAF"):
            toks = s.split()[1:]
            is_leaf = [int(x) for x in toks]
            stage = None
            continue
        if s.startswith("#UNLEAF"):
            toks = s.split()[1:]
            unleaf = [int(x) for x in toks]
            stage = None
            continue

        # 追加: delta dump（もし C++ 側で出しているなら拾う）
        # 例: "#DELTA 0.2 -0.1 0 0 ..."
        if s.startswith("#DELTA"):
            toks = s.split()[1:]
            delta = _parse_float_list(toks)
            stage = None
            continue

        # その他コメント
        if s.startswith("#"):
            stage = None
            continue

        if stage == "EDGE":
            a, b = s.split()
            edges.append((int(a), int(b)))

    if N is None or root is None:
        raise RuntimeError("dump block missing #PACKED_NODES or #ROOT")

    # 長さが足りない場合は補完（最低限動かす）
    if not is_leaf:
        # 子なしを葉とみなす（EDGEから推定）
        children = [[] for _ in range(N)]
        for p, q in edges:
            children[p].append(q)
        is_leaf = [1 if len(children[i]) == 0 else 0 for i in range(N)]

    if not unleaf:
        unleaf = [-1] * N

    if delta is not None and len(delta) != N:
        # 形式が違う（改行分割など）ならここでは捨てる
        delta = None

    return DumpBlock(
        idx=idx if idx >= 0 else 0,
        doc_id=doc_id,
        dump_seed=dump_seed,
        N=N,
        root=root,
        edges=edges,
        is_leaf=is_leaf,
        unleaf=unleaf,
        delta=delta,
    )


# ============================================================
# 4) Algorithm A（1D tree embedding）実装
# ============================================================

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


def algoA_embed(root: int, children: List[List[int]], is_leaf: List[int]) -> Tuple[Dict[int, int], int, List[int]]:
    """
    擬似コードに忠実に：
      f = 0 初期化
      treeEmbed(T, k=0) を呼び、葉だけ f[u] に 1D座標を設定して返す
    """
    n = len(children)
    f = {u: 0 for u in range(n)}
    depth, H = compute_depths(root, children)

    def add_on_leaves(u: int, delta: int):
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
            delta = kprime - k
            k = kprime
            if i > 0:
                gap = max(delta_old, delta)
                add_on_leaves(c, gap)
                k += gap
            if i != imax:
                k += 1
            delta_old = delta
            i += 1
        return k

    treeEmbed(root, 0)

    # reachable only
    reachable = [0] * n
    stack = [root]
    reachable[root] = 1
    while stack:
        u = stack.pop()
        for v in children[u]:
            if not reachable[v]:
                reachable[v] = 1
                stack.append(v)

    f2 = {u: f[u] for u in f if is_leaf[u] and reachable[u]}
    return f2, H, depth


# ============================================================
# 5) + / - split（dump の delta があればそれで）
# ============================================================

def split_by_delta(
    f_leafnode: Dict[int, int],
    unleaf: List[int],
    delta: Optional[List[float]],
    eps: float = 1e-12,
) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, float], bool]:
    """
    Returns:
      plus_vids, minus_vids,
      sign_map: vid -> (+1|-1),
      mass_map: vid -> |delta|,
      sign_dump_available: bool
    """
    sign_map: Dict[int, int] = {}
    mass_map: Dict[int, float] = {}

    if delta is None:
        return [], [], sign_map, mass_map, False

    plus: List[int] = []
    minus: List[int] = []
    for leaf_node, x in f_leafnode.items():
        vid = unleaf[leaf_node] if 0 <= leaf_node < len(unleaf) else -1
        if vid < 0:
            continue
        d = float(delta[leaf_node])
        if abs(d) <= eps:
            continue
        sg = 1 if d > 0 else -1
        sign_map[int(vid)] = sg
        mass_map[int(vid)] = abs(d)
        if sg > 0:
            plus.append(int(vid))
        else:
            minus.append(int(vid))

    return plus, minus, sign_map, mass_map, True


# ============================================================
# 6) Algorithm B（1D 最近傍 scan）+ brute check
# ============================================================

def sort_pts(pts: Iterable[Pt]) -> List[Pt]:
    return sorted(list(pts), key=lambda p: (p.x, p.id))

def abs_i(a: int) -> int:
    return a if a >= 0 else -a

def nn_scan_algoB_pseudocode(X: List[Pt], Y: List[Pt], tie_break: str = "left") -> List[NNResult]:
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
        # 論文と同じ条件式（strict <）
        while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) < abs_i(Ys[j].x - u.x)):
            j += 1

        v = Ys[j]
        d = abs_i(v.x - u.x)

        # 便宜上の right tie-break
        if tie_break == "right":
            while (j + 1 < m) and (abs_i(Ys[j + 1].x - u.x) == abs_i(Ys[j].x - u.x)):
                j += 1
                v = Ys[j]
                d = abs_i(v.x - u.x)

        res.append(NNResult(u.id, u.x, v.id, v.x, d))

    return res


def nn_bruteforce(X: List[Pt], Y: List[Pt], tie_break: str = "left") -> List[NNResult]:
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
            return False, f"u={u} で不一致: scan->(v={a.v_id},d={a.d1},vx={a.v_x}) brute->(v={b.v_id},d={b.d1},vx={b.v_x})"
    return True, "OK"


def print_results(title: str, rr: List[NNResult]):
    print(title)
    print(" u_id  u_x   ->   v_id  v_x   d1")
    for r in rr:
        print(f"{r.u_id:>6d} {r.u_x:>4d}        {r.v_id:>6d} {r.v_x:>4d} {r.d1:>4d}")
    print()


# ============================================================
# 7) 埋め込み out の保存（あなたの旧 .out を拡張）
# ============================================================

def write_embedding_out(
    out_path: str,
    doc_id: int,
    idx: int,
    seed: int,
    packedN: int,
    H: int,
    items: List[Dict[str, Any]],  # {"name","pid","vid","x","depth","sign","mass"}
    plus_cnt: Optional[int] = None,
    minus_cnt: Optional[int] = None,
    sign_dump: bool = False,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as g:
        g.write(f"=== doc_id={doc_id} idx={idx} seed={seed} ===\n\n")
        g.write(f"packedN={packedN} max_depth(H)={H}\n")
        if plus_cnt is not None and minus_cnt is not None:
            g.write(f"#leaves={len(items)}  plus={plus_cnt} minus={minus_cnt}  (sign_dump={'YES' if sign_dump else 'NO'})\n\n")
        else:
            g.write(f"#leaves={len(items)}\n\n")

        g.write("葉(左→右): name, id(packed), vocab_id, x, depth, sign, mass\n")

        items2 = sorted(items, key=lambda d: (int(d["x"]), int(d["vid"])))
        for it in items2:
            name = str(it.get("name", f"v{it['vid']}"))
            pid  = int(it.get("pid", -1))
            vid  = int(it["vid"])
            x    = int(it["x"])
            dep  = int(it.get("depth", -1))
            sg   = str(it.get("sign", "?"))
            ms   = it.get("mass", None)
            if ms is None:
                g.write(f" {name:>8s}  (id={pid:3d})  vid={vid:6d}  x={x:4d}  depth={dep:2d}  sign={sg}\n")
            else:
                g.write(f" {name:>8s}  (id={pid:3d})  vid={vid:6d}  x={x:4d}  depth={dep:2d}  sign={sg}  mass={float(ms):.6f}\n")


# ============================================================
# 8) ot_estimators 呼び出し（dump生成）
# ============================================================

def import_ot_estimators(so_dir: str):
    if so_dir not in sys.path:
        sys.path.append(so_dir)
    import ot_estimators  # type: ignore
    return ot_estimators


def call_load_vocabulary(ot, vocab: np.ndarray, seed: int):
    # あなたのログでは seed 引数付きに改造済み想定だが、互換のため fallback を用意
    try:
        ot.load_vocabulary(vocab, seed)
        print(f"[load_vocabulary] called with seed={seed}")
    except TypeError:
        ot.load_vocabulary(vocab)
        print(f"[load_vocabulary] called (no-seed signature) seed={seed} ignored")


def call_configure_dump(ot, dump_out: str, dump_limit: int, dump_sign: bool = True, dump_seed: Optional[int] = None):
    """
    configure_flowtree_dump のシグネチャが環境で違う可能性があるので try を重ねる。
    あなたの dir(ot) に 'configure_flowtree_dump' が見えている前提。
    """
    fn = getattr(ot, "configure_flowtree_dump", None)
    if fn is None:
        raise RuntimeError("ot_estimators.OTEstimators has no configure_flowtree_dump")

    # ありそうなパターンを順に試す
    tried = []
    for args in [
        (dump_out, dump_limit, dump_sign, dump_seed),
        (dump_out, dump_limit, dump_sign),
        (dump_out, dump_limit),
        (dump_out,),
    ]:
        try:
            fn(*args)
            return
        except TypeError as e:
            tried.append((args, str(e)))
            continue

    msg = "\n".join([f"  tried {a} -> {err}" for a, err in tried])
    raise RuntimeError("configure_flowtree_dump signature mismatch:\n" + msg)


def run_flowtree_rank_for_docs(ot, query: List[Tuple[int, float]], docs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    flowtree_rank(query, input_ids, output_ids, output_scores, to_sort)
    を想定して呼ぶ。dump を出すのが主目的だが、scoreも返す。
    """
    input_ids = np.array(docs, dtype=np.int32)
    output_ids = np.empty_like(input_ids)
    output_scores = np.empty((len(docs),), dtype=np.float32)

    # 互換性のために to_sort 引数の有無を吸収
    try:
        ot.flowtree_rank(query, input_ids, output_ids, output_scores, False)
    except TypeError:
        ot.flowtree_rank(query, input_ids, output_ids, output_scores)

    return output_ids, output_scores


# ============================================================
# 9) main
# ============================================================

def parse_docs_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="otdata dir containing vocab.npy, dataset.npy, queries.npy")
    ap.add_argument("--so_dir", required=True, help="dir containing ot_estimators*.so")
    ap.add_argument("--seed", type=int, default=0, help="seed for FlowTree random shift (0 allowed)")
    ap.add_argument("--qid", type=int, default=0, help="query index into queries.npy")
    ap.add_argument("--docs", default="0,1,2", help="comma-separated doc ids (indices into dataset)")
    ap.add_argument("--dump_out", required=True, help="path to save packed subtree dump txt")
    ap.add_argument("--dump_limit", type=int, default=3, help="number of dumps to write")
    ap.add_argument("--emb_out_dir", default="", help="if set, save 1D embedding .out per doc into this directory")
    ap.add_argument("--assume_docs_in_order", action="store_true",
                    help="if dump block doesn't contain doc_id, assume idx corresponds to docs list order")
    ap.add_argument("--tie", choices=["left", "right"], default="left", help="tie-break for NN scan")
    ap.add_argument("--print", dest="do_print", action="store_true", help="print NN tables")
    ap.add_argument("--check", action="store_true", help="check scan vs brute")
    args = ap.parse_args()

    data_dir = args.data_dir
    vocab_path = os.path.join(data_dir, "vocab.npy")
    dataset_path = os.path.join(data_dir, "dataset.npy")
    queries_path = os.path.join(data_dir, "queries.npy")

    docs = parse_docs_list(args.docs)
    if not docs:
        raise RuntimeError("docs is empty")
    seed = int(args.seed)

    # ---- load data ----
    vocab = load_vocab(vocab_path)
    dataset = load_object_measures(dataset_path)
    queries = load_object_measures(queries_path)

    if args.qid < 0 or args.qid >= len(queries):
        raise RuntimeError(f"qid out of range: {args.qid} (queries={len(queries)})")
    query = queries[args.qid]

    # ---- load ot_estimators ----
    ot_estimators = import_ot_estimators(args.so_dir)
    ot = ot_estimators.OTEstimators()

    # ---- configure & build ----
    call_load_vocabulary(ot, vocab, seed)
    ot.load_dataset(dataset)

    # dump設定（C++改造済み想定）
    call_configure_dump(ot, args.dump_out, args.dump_limit, dump_sign=True, dump_seed=seed)

    # ---- run rank (to generate dumps) ----
    out_ids, out_scores = run_flowtree_rank_for_docs(ot, query, docs)
    print(f"[flowtree_rank] ran for docs={docs} -> dump={args.dump_out}")

    # ---- parse dumps ----
    blocks = parse_multi_dump(args.dump_out)
    if len(blocks) == 0:
        raise RuntimeError("no dump blocks parsed")

    # dumpブロックを doc_id で引けるように
    # doc_id が無い場合は idx と docs順で対応づけ
    block_by_doc: Dict[int, DumpBlock] = {}

    for b in blocks:
        if b.doc_id is not None:
            block_by_doc[int(b.doc_id)] = b

    # fallback mapping
    if not block_by_doc and args.assume_docs_in_order:
        for i, b in enumerate(blocks):
            if i < len(docs):
                block_by_doc[int(docs[i])] = b

    # ---- per-doc: embed + split + NN ----
    for k, doc_id in enumerate(docs):
        if doc_id not in block_by_doc:
            # 見つからない場合は idx順で拾う（最後の保険）
            if k < len(blocks):
                b = blocks[k]
            else:
                print(f"[warn] no dump block for doc_id={doc_id}, skipped")
                continue
        else:
            b = block_by_doc[doc_id]

        # tree structure
        children = [[] for _ in range(b.N)]
        for p, q in b.edges:
            if 0 <= p < b.N and 0 <= q < b.N:
                children[p].append(q)
        for ch in children:
            ch.sort()

        # embed
        f_leafnode, H, depth = algoA_embed(b.root, children, b.is_leaf)

        # split by delta (+/-)
        plus_vids, minus_vids, sign_map, mass_map, sign_dump = split_by_delta(
            f_leafnode=f_leafnode,
            unleaf=b.unleaf,
            delta=b.delta,
        )

        # leaf items for saving
        leaf_items: List[Dict[str, Any]] = []
        for leaf_node, x in f_leafnode.items():
            vid = b.unleaf[leaf_node] if 0 <= leaf_node < len(b.unleaf) else -1
            if vid < 0:
                continue
            sg = sign_map.get(int(vid), 0)
            leaf_items.append({
                "name": f"v{int(vid)}",
                "pid": int(leaf_node),
                "vid": int(vid),
                "x": int(x),
                "depth": int(depth[leaf_node]) if 0 <= leaf_node < len(depth) else -1,
                "sign": "+" if sg > 0 else ("-" if sg < 0 else "?"),
                "mass": mass_map.get(int(vid), None),
            })

        print(f"=== doc_id={doc_id} idx={b.idx} dump_seed={b.dump_seed if b.dump_seed is not None else -1} packedN={b.N} H={H} ===")
        if sign_dump:
            print(f"[split sign] plus={len(plus_vids)} minus={len(minus_vids)}  (sign_dump=YES)")
        else:
            print(f"[split sign] sign_dump=NO (no #DELTA found)")

        # save embedding out
        if args.emb_out_dir:
            out_path = os.path.join(args.emb_out_dir, f"doc{doc_id:03d}_idx{b.idx:03d}_seed{seed}.out")
            write_embedding_out(
                out_path=out_path,
                doc_id=doc_id,
                idx=b.idx,
                seed=seed,
                packedN=b.N,
                H=H,
                items=leaf_items,
                plus_cnt=len(plus_vids) if sign_dump else None,
                minus_cnt=len(minus_vids) if sign_dump else None,
                sign_dump=sign_dump,
            )

        # NN only if sign dump available
        if not sign_dump:
            continue
        if len(plus_vids) == 0 or len(minus_vids) == 0:
            print("[warn] plus or minus empty, skipped NN")
            continue

        # build vid -> x map from embedding
        # f_leafnode is leaf_node -> x, so convert using unleaf
        vid_to_x: Dict[int, int] = {}
        for leaf_node, x in f_leafnode.items():
            vid = b.unleaf[leaf_node]
            if vid >= 0:
                vid_to_x[int(vid)] = int(x)

        X_pts = [Pt(v, vid_to_x[v]) for v in plus_vids if v in vid_to_x]
        Y_pts = [Pt(v, vid_to_x[v]) for v in minus_vids if v in vid_to_x]

        scan = nn_scan_algoB_pseudocode(X_pts, Y_pts, tie_break=args.tie)
        if args.do_print:
            print_results("=== + -> -  scan NN ===", scan)

        if args.check:
            brute = nn_bruteforce(X_pts, Y_pts, tie_break=args.tie)
            ok, msg = check_equal(scan, brute)
            print("[check +->-]", msg)
            if not ok:
                Xs = sort_pts(X_pts)
                Ys = sort_pts(Y_pts)
                print("X(sorted):", [(p.id, p.x) for p in Xs])
                print("Y(sorted):", [(p.id, p.x) for p in Ys])

        print()

    print("[done]")


if __name__ == "__main__":
    main()
