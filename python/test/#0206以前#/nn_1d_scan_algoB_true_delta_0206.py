#!/usr/bin/env python3
# nn_1d_scan_algoB_true_delta_0206.py
#
# 目的：
#  1) adj_001.out から packed木に出た S（語彙10個）を読み取る
#  2) query.npz から Q（語彙5個 + 重み）を読み取る
#  3) D = S \ Q（候補側5語）を作る
#  4) dataset.npz を走査し，「語彙集合が D と一致する文書」を探し，その重み d_w を取得
#  5) delta(id) = q_w(id) - d_w(id) を語彙ごとに計算（Q∪D を対象）
#  6) delta>0（供給）と delta<0（需要）に分けて，1D上の AlgoB（ワンパス相殺）を実行
#
# 注意：
#  - ここでの1D距離は |x_u-x_v|（埋め込み座標差）をコストとして使います（動作確認用）
#  - 「完全にFlowtreeと同じ」ではなく，「deltaの符号で相殺する」部分の確認が主目的です

import os
import re
import argparse
import numpy as np

# -------------------------
# embed1d_algoA_demo.py の .out パース
# 例: v47  (id= 0)  x=  0  depth=2
# -------------------------
_LEAF_RE = re.compile(
    r'^\s*(?P<name>v\d+)\s+\(id=\s*(?P<id>\d+)\)\s+x=\s*(?P<x>-?\d+)\s+depth=\s*(?P<depth>-?\d+)\s*$'
)

def parse_embed_out(path: str):
    f = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            m = _LEAF_RE.match(line.rstrip("\n"))
            if not m:
                continue
            name = m.group("name")
            x = int(m.group("x"))
            vid = int(name[1:])
            f[vid] = x
    if not f:
        raise RuntimeError(f"no v* leaves in {path}")
    return f

# -------------------------
# dataset.npz の (ids,weights,offsets) から doc を1つ取り出す
# -------------------------
def load_doc_from_flat(ids_flat, w_flat, offs, doc_id: int):
    s = int(offs[doc_id])
    e = int(offs[doc_id + 1])
    ids = ids_flat[s:e].astype(np.int32)
    ws  = w_flat[s:e].astype(np.float64)
    return ids, ws

# -------------------------
# AlgoB風：x昇順で(+側)と(-側)を2ポインタで相殺
# コスト = |x_u - x_v| * flow
# -------------------------
def algoB_scan_match(pos_ids, pos_mass, neg_ids, neg_mass, x_map):
    """
    pos_ids: delta>0 の語彙ID
    neg_ids: delta<0 の語彙ID
    pos_mass[u] = delta(u) > 0
    neg_mass[v] = -delta(v) > 0 （需要量として正にして渡す）
    """
    P = sorted([(x_map[u], u) for u in pos_ids], key=lambda t: t[0])
    N = sorted([(x_map[v], v) for v in neg_ids], key=lambda t: t[0])

    remP = {u: float(pos_mass[u]) for _, u in P}
    remN = {v: float(neg_mass[v]) for _, v in N}

    i = 0
    j = 0
    matches = []
    cost = 0.0

    while i < len(P) and j < len(N):
        xu, u = P[i]
        xv, v = N[j]

        fu = remP[u]
        fv = remN[v]
        if fu <= 1e-12:
            i += 1
            continue
        if fv <= 1e-12:
            j += 1
            continue

        flow = fu if fu < fv else fv
        dist1d = abs(xu - xv)

        remP[u] -= flow
        remN[v] -= flow

        matches.append((u, v, flow, dist1d))
        cost += flow * dist1d

        if remP[u] <= 1e-12:
            i += 1
        if remN[v] <= 1e-12:
            j += 1

    # 残りが出たら注意（普通は合計が一致していれば残らない）
    rem_pos = sum(max(0.0, remP[u]) for u in remP)
    rem_neg = sum(max(0.0, remN[v]) for v in remN)
    return matches, cost, rem_pos, rem_neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="tmp_flowtree_demo")
    ap.add_argument("--A_out", required=True, help="emb_A/adj_001.out")
    ap.add_argument("--B_out", required=True, help="emb_B/adj_001.out")
    ap.add_argument("--doc_id", type=int, default=None,
                    help="候補文書IDを手で指定したい場合（指定がなければD集合一致で検索）")
    ap.add_argument("--max_scan", type=int, default=5000,
                    help="dataset走査上限（demoは5000なのでそのままでOK）")
    args = ap.parse_args()

    data_dir = args.data_dir
    fA = parse_embed_out(args.A_out)
    fB = parse_embed_out(args.B_out)

    # packed木に出てきた語彙集合（S）
    S = sorted(set(fA.keys()) & set(fB.keys()))
    if len(S) == 0:
        raise RuntimeError("no common leaves between A_out and B_out")
    print(f"[S] size={len(S)}  S={S}")

    # query
    qz = np.load(os.path.join(data_dir, "query.npz"))
    Q_ids = [int(t) for t in qz["ids"].tolist()]
    Q_w   = [float(t) for t in qz["weights"].tolist()]
    q_map = {u: w for u, w in zip(Q_ids, Q_w)}
    print(f"[Q] ids={Q_ids}")
    print(f"[Q] weights={Q_w}  sum={sum(Q_w):.6f}")

    # 候補側語彙（D = S \ Q）
    D = sorted(set(S) - set(Q_ids))
    print(f"[D] size={len(D)}  D={D}")

    # dataset.npz
    dz = np.load(os.path.join(data_dir, "dataset.npz"))
    ids_flat = dz["ids"]
    w_flat   = dz["weights"]
    offs     = dz["offsets"]
    num_docs = len(offs) - 1
    print(f"[dataset] num_docs={num_docs}")

    # --- 候補文書を特定する ---
    target_doc = args.doc_id
    d_map = None

    if target_doc is not None:
        if not (0 <= target_doc < num_docs):
            raise RuntimeError(f"doc_id out of range: {target_doc}")
        ids, ws = load_doc_from_flat(ids_flat, w_flat, offs, target_doc)
        if set(ids.tolist()) != set(D):
            print("[warn] 指定doc_idの語彙集合がDと一致していません")
            print(" doc vocab =", sorted(set(ids.tolist())))
        d_map = {int(i): float(w) for i, w in zip(ids.tolist(), ws.tolist())}
    else:
        Dset = set(D)
        found = []
        scanN = min(args.max_scan, num_docs)
        for doc_id in range(scanN):
            ids, ws = load_doc_from_flat(ids_flat, w_flat, offs, doc_id)
            if set(ids.tolist()) == Dset:
                found.append(doc_id)
                # 最初の1個で十分ならbreakしてOKだが、複数あるかもなので一応2つまで探す
                if len(found) >= 2:
                    break

        if not found:
            raise RuntimeError("D集合と完全一致する文書が見つかりませんでした（衝突/不一致の可能性）")
        target_doc = found[0]
        ids, ws = load_doc_from_flat(ids_flat, w_flat, offs, target_doc)
        d_map = {int(i): float(w) for i, w in zip(ids.tolist(), ws.tolist())}

        print(f"[doc] found doc_id={target_doc}  (also_found={found[1:]})")

    # 文書側重みチェック
    d_sum = sum(d_map.get(v, 0.0) for v in D)
    print(f"[doc] D weights sum={d_sum:.6f}")
    print("[doc] pairs:", [(v, d_map[v]) for v in D])

    # --- delta の構築 ---
    # 対象は S（=Q∪D のはず）だけで十分
    delta = {}
    for v in S:
        qw = q_map.get(v, 0.0)     # クエリに無ければ0
        dw = d_map.get(v, 0.0)     # 文書に無ければ0
        delta[v] = qw - dw

    # delta の符号で分割
    pos = [v for v in S if delta[v] >  1e-12]   # 供給
    neg = [v for v in S if delta[v] < -1e-12]   # 需要
    pos_mass = {v: delta[v] for v in pos}
    neg_mass = {v: -delta[v] for v in neg}      # 需要量は正にして持つ

    sum_pos = sum(pos_mass.values())
    sum_neg = sum(neg_mass.values())
    print("\n=== delta summary ===")
    print("delta (v: q - d):")
    for v in S:
        print(f"  v{v:<3d}  q={q_map.get(v,0.0):.6f}  d={d_map.get(v,0.0):.6f}  delta={delta[v]:+.6f}")
    print(f"pos={pos}  sum_pos={sum_pos:.6f}")
    print(f"neg={neg}  sum_neg={sum_neg:.6f}")

    # 合計がズレる場合の注意（浮動小数誤差くらいならOK）
    if abs(sum_pos - sum_neg) > 1e-6:
        print("[warn] sum_pos != sum_neg。残りが出る可能性があります（データや丸めの問題）")

    # --- AlgoB 実行（A座標とB座標の両方で）---
    def run_one(tag, xmap):
        print(f"\n=== AlgoB scan using {tag} x ===")
        if len(pos) == 0 or len(neg) == 0:
            print("pos or neg is empty; nothing to match.")
            return
        matches, cost, rem_pos, rem_neg = algoB_scan_match(pos, pos_mass, neg, neg_mass, xmap)
        for (u, v, flow, dist1d) in matches:
            print(f"  (+)v{u:<3d}  (-)v{v:<3d}  flow={flow:.6f}  |x_u-x_v|={dist1d}")
        print(f"total_1d_cost={cost:.6f}  rem_pos={rem_pos:.6e}  rem_neg={rem_neg:.6e}")

    run_one("A_out", fA)
    run_one("B_out", fB)

    print("\n[done]")
    print(f"doc_id used = {target_doc}")

if __name__ == "__main__":
    main()
