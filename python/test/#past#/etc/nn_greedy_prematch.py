#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_greedy_prematch.py

目的:
  FlowTree の 1D NN ログ（u -> v, d1）と、葉dumpに含まれる sign/mass(|delta|) を使って
  d1 昇順に greedy に「プレマッチで流せるだけ流す」処理を行い、
  どの (u,v) にどれだけ流したかを可視化する。

前提:
  - data_dir に vocab.npy がある（GloVeベクトル）
  - nnログ（flowtree_real_1d_nn_pipeline.py の --print 出力を保存したもの等）から
    doc_id ブロック内の「u_id u_x -> v_id v_x d1」表を読める
  - さらに sign/mass は以下のどちらかから取る
    A) embファイル（--emb_file）: 1行に vid, x, depth, sign, mass が含まれる形式
    B) dump/log（--log）内に葉一覧があり、そこに sign, mass が含まれる形式
       （あなたの旧 out 形式 + sign/mass 追記のイメージ）

使い方例:
  # 1) まず flowtree_real_1d_nn_pipeline.py の標準出力を log に保存しておく（例）
  #    python3 ... --print ... > /mnt/c/.../run_seed111.log

  # 2) その log と、同時に出した emb を使って greedy prematch
  python3 nn_greedy_prematch.py \
    --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
    --log "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed111.log" \
    --doc_id 0 \
    --emb_file "/mnt/c/Users/成見/0905_work/tmp_real_dump/emb_real/emb_doc000_seed111.txt" \
    --show_top 200 \
    --out_json "/mnt/c/Users/成見/0905_work/tmp_real_dump/prematch_doc0_seed111.json"

  # emb_file が分からない場合: emb_out_dir から doc_id/seed を含むファイルを自動探索
  python3 nn_greedy_prematch.py \
    --data_dir "..." --log "..." --doc_id 0 \
    --emb_out_dir "/mnt/c/Users/成見/0905_work/tmp_real_dump/emb_real" --seed 111
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# パース用
# ----------------------------

DOC_HDR_RE = re.compile(r"^===\s*doc_id=(\d+)\b.*?===", re.M)

# 例: "   50     0          246     2     2"
NN_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)

# emb/葉dump想定:
# 例: " v47  (id= 0)  vid=    50  x=  0  depth=5  sign=+1  mass=0.123456"
LEAF_LINE_RE = re.compile(
    r"vid\s*=\s*(\d+).*?\bx\s*=\s*(-?\d+).*?\bdepth\s*=\s*(-?\d+).*?\bsign\s*=\s*([+\-]?\d+|[+\-])"
    r"(?:.*?\bmass\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+\-]?\d+)?))?",
    re.I,
)

# dump内の簡易 sign/mass 形式が別なら、ここに追加で吸収しても良い


@dataclass
class NNCand:
    u_vid: int
    u_x: int
    v_vid: int
    v_x: int
    d1: int
    l2: float = float("nan")


@dataclass
class FlowMatch:
    order: int          # d1順での処理順
    u_vid: int
    v_vid: int
    d1: int
    l2: float
    flow: float
    rem_u_after: float
    rem_v_after: float


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_doc_block(text: str, doc_id: int) -> str:
    """
    log から doc_id のブロック文字列を切り出す。
    """
    matches = list(DOC_HDR_RE.finditer(text))
    if not matches:
        raise ValueError("doc header (=== doc_id=... ===) not found in log")

    # header開始位置の配列
    spans = []
    for m in matches:
        did = int(m.group(1))
        spans.append((did, m.start(), m.end()))

    # doc_id の開始点を探す
    start = None
    end = None
    for i, (did, s, e) in enumerate(spans):
        if did == doc_id:
            start = s
            # 次のヘッダまで
            end = spans[i + 1][1] if (i + 1) < len(spans) else len(text)
            break
    if start is None:
        raise ValueError(f"doc_id={doc_id} block not found in log")

    return text[start:end]


def parse_nn_rows(block: str) -> List[NNCand]:
    """
    docブロックから NN表をパース。
    注意: 同じブロックに表が2つあっても、ここは全部拾う。
    後で重複排除＆意味のある集合にする。
    """
    rows = NN_ROW_RE.findall(block)
    cands: List[NNCand] = []
    for u_vid, u_x, v_vid, v_x, d1 in rows:
        cands.append(
            NNCand(
                u_vid=int(u_vid),
                u_x=int(u_x),
                v_vid=int(v_vid),
                v_x=int(v_x),
                d1=int(d1),
            )
        )
    # u_vid が同じ行が複数表で出てきうるので、最後に整理する
    return cands


def dedupe_by_u_keep_first(cands: List[NNCand]) -> List[NNCand]:
    """
    u_vid ごとに最初の出現を採用（log-order の表を優先したい場合に自然）。
    もし d1-sorted 表を優先したい場合は、入力をその表だけに絞るのが良い。
    """
    seen = set()
    out = []
    for c in cands:
        if c.u_vid in seen:
            continue
        seen.add(c.u_vid)
        out.append(c)
    return out


def parse_leaf_sign_mass_from_text(text: str) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    text(emb/dump) から vid -> sign, vid -> mass を抜く。
    mass が無い場合は 0 として扱う（=流せないので実質無効）。
    """
    sign_map: Dict[int, int] = {}
    mass_map: Dict[int, float] = {}

    for line in text.splitlines():
        m = LEAF_LINE_RE.search(line)
        if not m:
            continue
        vid = int(m.group(1))
        # x, depth は今回不要（欲しければ拡張）
        sraw = m.group(4).strip()
        if sraw in ["+", "+1", "1"]:
            s = +1
        elif sraw in ["-", "-1"]:
            s = -1
        else:
            # "2" など想定外は signとして読めないのでスキップ
            try:
                s = int(sraw)
                s = +1 if s > 0 else -1
            except Exception:
                continue

        mass = 0.0
        if m.group(5) is not None:
            try:
                mass = float(m.group(5))
            except Exception:
                mass = 0.0

        sign_map[vid] = s
        mass_map[vid] = mass

    return sign_map, mass_map


def find_emb_file(emb_out_dir: str, doc_id: int, seed: int) -> str:
    """
    emb_out_dir から doc_id/seed っぽいファイルを探す（緩め）。
    """
    patterns = [
        os.path.join(emb_out_dir, f"*doc*{doc_id}*seed*{seed}*.txt"),
        os.path.join(emb_out_dir, f"*doc*{doc_id:03d}*seed*{seed}*.txt"),
        os.path.join(emb_out_dir, f"*{doc_id}*{seed}*.txt"),
        os.path.join(emb_out_dir, f"*{doc_id:03d}*{seed}*.txt"),
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(
            f"emb file not found in emb_out_dir={emb_out_dir} for doc_id={doc_id}, seed={seed}\n"
            f"tried patterns: {patterns}"
        )
    # 最新更新のものを優先
    hits.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return hits[0]


def l2_dist(vocab: np.ndarray, a: int, b: int) -> float:
    # vocab は float32 (V,D)
    diff = vocab[a] - vocab[b]
    return float(np.sqrt(np.dot(diff, diff)))


# ----------------------------
# greedy prematch
# ----------------------------

def greedy_prematch(
    cands: List[NNCand],
    sign_map: Dict[int, int],
    mass_map: Dict[int, float],
    vocab: np.ndarray,
    sort_by: str = "d1",
) -> Tuple[List[FlowMatch], Dict[int, float], Dict[int, float], Dict[str, int]]:
    """
    cands: u->v 候補（基本 + -> - を想定）
    sign_map/mass_map: vid -> sign, |delta|
    sort_by: "d1"（基本）/ "l2" も可
    """
    # plus/minus 残量
    rem_plus: Dict[int, float] = {}
    rem_minus: Dict[int, float] = {}
    for vid, s in sign_map.items():
        m = float(mass_map.get(vid, 0.0) or 0.0)
        if m <= 0:
            continue
        if s > 0:
            rem_plus[vid] = m
        else:
            rem_minus[vid] = m

    # 候補の L2 を埋める＋符号チェック
    filtered: List[NNCand] = []
    stat = {
        "cand_total": 0,
        "cand_kept": 0,
        "cand_skip_sign_missing": 0,
        "cand_skip_not_plus_minus": 0,
    }
    for c in cands:
        stat["cand_total"] += 1
        su = sign_map.get(c.u_vid, None)
        sv = sign_map.get(c.v_vid, None)
        if su is None or sv is None:
            stat["cand_skip_sign_missing"] += 1
            continue
        if not (su > 0 and sv < 0):
            stat["cand_skip_not_plus_minus"] += 1
            continue
        c.l2 = l2_dist(vocab, c.u_vid, c.v_vid)
        filtered.append(c)
        stat["cand_kept"] += 1

    # ソート
    if sort_by == "l2":
        filtered.sort(key=lambda x: (x.l2, x.d1, x.u_vid, x.v_vid))
    else:
        # 基本は d1 で優先順位
        filtered.sort(key=lambda x: (x.d1, x.l2, x.u_vid, x.v_vid))

    matches: List[FlowMatch] = []
    skip_u0 = 0
    skip_v0 = 0
    used_rows = 0

    for i, c in enumerate(filtered):
        ru = rem_plus.get(c.u_vid, 0.0)
        rv = rem_minus.get(c.v_vid, 0.0)
        if ru <= 0:
            skip_u0 += 1
            continue
        if rv <= 0:
            skip_v0 += 1
            continue
        t = ru if ru < rv else rv
        if t <= 0:
            continue
        rem_plus[c.u_vid] = ru - t
        rem_minus[c.v_vid] = rv - t
        used_rows += 1
        matches.append(
            FlowMatch(
                order=len(matches) + 1,
                u_vid=c.u_vid,
                v_vid=c.v_vid,
                d1=c.d1,
                l2=c.l2,
                flow=t,
                rem_u_after=rem_plus[c.u_vid],
                rem_v_after=rem_minus[c.v_vid],
            )
        )

    stat.update(
        {
            "rows_used": used_rows,
            "rows_skipped_u_exhausted": skip_u0,
            "rows_skipped_v_exhausted": skip_v0,
            "matched_edges": len(matches),
        }
    )
    return matches, rem_plus, rem_minus, stat


def print_matches(matches: List[FlowMatch], top: int = 200) -> None:
    print("\n[matched] d1-priority greedy flows (show_top={})".format(top))
    print(" ord   u_vid      v_vid     d1        l2        flow     rem_u     rem_v")
    print("-------------------------------------------------------------------------")
    for m in matches[:top]:
        print(
            f"{m.order:4d} {m.u_vid:7d} {m.v_vid:9d} {m.d1:6d} "
            f"{m.l2:9.6f} {m.flow:10.6f} {m.rem_u_after:9.6f} {m.rem_v_after:9.6f}"
        )
    if len(matches) > top:
        print(f"... ({len(matches)-top} more)")


def summarize_remaining(rem_plus: Dict[int, float], rem_minus: Dict[int, float]) -> None:
    splus = sum(max(0.0, v) for v in rem_plus.values())
    sminus = sum(max(0.0, v) for v in rem_minus.values())
    nz_plus = sum(1 for v in rem_plus.values() if v > 1e-12)
    nz_minus = sum(1 for v in rem_minus.values() if v > 1e-12)
    print("\n[remaining residual]")
    print(f"  sum(rem_plus)  = {splus:.6f}   (#nonzero={nz_plus})")
    print(f"  sum(rem_minus) = {sminus:.6f}   (#nonzero={nz_minus})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="otdata_glove50_full directory")
    ap.add_argument("--log", required=True, help="NN log file (run_seedXXX.log etc)")
    ap.add_argument("--doc_id", required=True, type=int, help="doc_id block to analyze")

    ap.add_argument("--emb_file", default=None, help="explicit emb file path (recommended)")
    ap.add_argument("--emb_out_dir", default=None, help="directory to auto-find emb file")
    ap.add_argument("--seed", type=int, default=None, help="seed for emb auto-find (required if emb_out_dir used)")

    ap.add_argument("--sort_by", choices=["d1", "l2"], default="d1", help="priority key (default: d1)")
    ap.add_argument("--show_top", type=int, default=200, help="rows to show in matched table")
    ap.add_argument("--out_json", default=None, help="write results JSON (optional)")
    ap.add_argument("--use_first_table_per_u", action="store_true",
                    help="dedupe by u_vid keeping first occurrence (useful if log contains 2 tables)")

    args = ap.parse_args()

    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.npy not found: {vocab_path}")

    # vocab は大きいので mmap
    vocab = np.load(vocab_path, mmap_mode="r")
    if vocab.ndim != 2 or vocab.dtype != np.float32:
        print(f"[warn] vocab dtype/shape unusual: shape={vocab.shape}, dtype={vocab.dtype}")

    text = read_text(args.log)
    block = extract_doc_block(text, args.doc_id)

    # NN候補
    cands = parse_nn_rows(block)
    if not cands:
        raise ValueError("No NN rows found in the doc block. Check log format.")
    if args.use_first_table_per_u:
        cands = dedupe_by_u_keep_first(cands)

    # sign/mass の取得
    emb_text = None
    emb_used = None
    if args.emb_file:
        emb_used = args.emb_file
        emb_text = read_text(args.emb_file)
    elif args.emb_out_dir:
        if args.seed is None:
            raise ValueError("--seed is required when using --emb_out_dir")
        emb_used = find_emb_file(args.emb_out_dir, args.doc_id, args.seed)
        emb_text = read_text(emb_used)
    else:
        # emb が無い場合: logブロック自体に sign/mass が含まれていればそこから取る
        emb_used = "(from log block)"
        emb_text = block

    sign_map, mass_map = parse_leaf_sign_mass_from_text(emb_text)
    if not sign_map:
        raise ValueError(
            "sign/mass parse failed (no leaf lines matched).\n"
            "Either pass --emb_file (recommended) or ensure your log contains leaf lines with 'vid=.. sign=.. mass=..'."
        )

    # greedy prematch
    matches, rem_plus, rem_minus, stat = greedy_prematch(
        cands=cands,
        sign_map=sign_map,
        mass_map=mass_map,
        vocab=vocab,
        sort_by=args.sort_by,
    )

    # レポート
    print(f"=== doc_id={args.doc_id} ===")
    print(f"[input] log={args.log}")
    print(f"[input] sign/mass source={emb_used}")
    print(f"[stats] cand_total={stat['cand_total']}  kept(+->-)={stat['cand_kept']}  "
          f"skip_sign_missing={stat['cand_skip_sign_missing']}  skip_not_plusminus={stat['cand_skip_not_plus_minus']}")
    print(f"[stats] matched_edges={stat['matched_edges']}  rows_used={stat['rows_used']}  "
          f"skipped(u_exhausted)={stat['rows_skipped_u_exhausted']}  skipped(v_exhausted)={stat['rows_skipped_v_exhausted']}")

    total_flow = sum(m.flow for m in matches)
    print(f"[summary] total_flow_sent = {total_flow:.6f}")

    print_matches(matches, top=args.show_top)
    summarize_remaining(rem_plus, rem_minus)

    # JSON 出力
    if args.out_json:
        out = {
            "doc_id": args.doc_id,
            "log": args.log,
            "sign_mass_source": emb_used,
            "sort_by": args.sort_by,
            "stats": stat,
            "total_flow_sent": total_flow,
            "matches": [
                {
                    "order": m.order,
                    "u_vid": m.u_vid,
                    "v_vid": m.v_vid,
                    "d1": m.d1,
                    "l2": m.l2,
                    "flow": m.flow,
                    "rem_u_after": m.rem_u_after,
                    "rem_v_after": m.rem_v_after,
                }
                for m in matches
            ],
            "remaining_plus": {str(k): float(v) for k, v in rem_plus.items() if v > 1e-12},
            "remaining_minus": {str(k): float(v) for k, v in rem_minus.items() if v > 1e-12},
        }
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[wrote] {args.out_json}")


if __name__ == "__main__":
    main()