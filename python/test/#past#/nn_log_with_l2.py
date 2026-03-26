#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_log_with_l2.py

目的:
  FlowTree 実行ログ（flowtree_real_1d_nn_pipeline.py の --print 出力など）
  に含まれる「=== + -> -  scan NN ===」の表を拾い、
  各ペア (u_id, v_id) について vocab.npy 上の L2 距離を計算して併記する。

要求対応:
  - vocab_words.npy は読まない（単語は出さない）
  - ログ順の表 + d1昇順の表 を両方出す
  - --nn_log を使ったコマンドに合わせる（--log の alias）
  - --doc_id で doc_id を絞れる
  - --top で行数制限
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------
# パース用
# ---------------------------

@dataclass
class PairRow:
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int
    l2: float


DOC_HDR_RE = re.compile(
    r"^===\s*doc_id=(\d+)\s+idx=(\d+)\s+dump_seed=([-\d]+)\s+packedN=(\d+)\s+H=(\d+)\s*===",
    re.M,
)
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)


def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def l2dist(vocab: np.ndarray, a: int, b: int) -> float:
    va = vocab[a]
    vb = vocab[b]
    d = va - vb
    return float(np.sqrt(np.dot(d, d)))


def parse_sections(text: str) -> List[Tuple[int, int, int, int, int, int]]:
    """
    doc header の開始位置を拾い、(doc_id, idx, dump_seed, packedN, H, start_pos) の list を返す
    """
    out = []
    for m in DOC_HDR_RE.finditer(text):
        doc_id = int(m.group(1))
        idx = int(m.group(2))
        dump_seed = int(m.group(3))
        packedN = int(m.group(4))
        H = int(m.group(5))
        start_pos = m.start()
        out.append((doc_id, idx, dump_seed, packedN, H, start_pos))

    # doc header が無いログでも落ちないように
    if not out:
        out.append((-1, -1, -1, -1, -1, 0))
    return out


def extract_nn_rows(block_text: str) -> List[Tuple[int, int, int, int, int]]:
    """
    ブロック中の NN 表の行を拾う（形式: u_id u_x v_id v_x d1）
    """
    rows = []
    for (u_id, u_x, v_id, v_x, d1) in ROW_RE.findall(block_text):
        rows.append((int(u_id), int(u_x), int(v_id), int(v_x), int(d1)))
    return rows


# ---------------------------
# 出力
# ---------------------------

def format_table(rows: List[PairRow], top: Optional[int] = None) -> str:
    if top is not None and top > 0:
        rows = rows[:top]

    lines = []
    lines.append(" u_id    u_x   ->    v_id    v_x    d1        l2")
    lines.append("--------------------------------------------------------")
    for r in rows:
        lines.append(
            f"{r.u_id:6d}  {r.u_x:5d}       {r.v_id:6d}  {r.v_x:5d}  {r.d1:4d}  {r.l2:10.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_dir",
        required=True,
        help="otdata_glove50_full ディレクトリ（vocab.npy が必要）",
    )

    # 互換: --log でも --nn_log でも受け付ける
    ap.add_argument(
        "--log",
        dest="log_path",
        default="",
        help="FlowTree 実行ログファイル（- で stdin）",
    )
    ap.add_argument(
        "--nn_log",
        dest="log_path",
        default="",
        help="(alias) FlowTree 実行ログファイル",
    )

    ap.add_argument("--out", default="", help="出力ファイル（未指定なら stdout）")
    ap.add_argument("--top", type=int, default=0, help="各表の上位行数（0なら全部）")
    ap.add_argument("--doc_id", type=int, default=None, help="指定した doc_id だけ出す（未指定なら全部）")

    args = ap.parse_args()

    if not args.log_path:
        ap.error("the following arguments are required: --log/--nn_log")

    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"vocab.npy not found: {vocab_path}")

    vocab = np.load(vocab_path, mmap_mode="r")  # shape (V, D) float32
    text = read_text(args.log_path)

    sections = parse_sections(text)

    # セクション境界
    section_ranges: List[Tuple[int, int, int, int, int, str]] = []
    for i, (doc_id, idx, dump_seed, packedN, H, start_pos) in enumerate(sections):
        end_pos = sections[i + 1][5] if i + 1 < len(sections) else len(text)
        section_ranges.append((doc_id, idx, dump_seed, packedN, H, text[start_pos:end_pos]))

    top = args.top if args.top and args.top > 0 else None

    out_lines: List[str] = []
    out_lines.append(f"[info] vocab: shape={vocab.shape} dtype={vocab.dtype}")
    out_lines.append(f"[info] log  : {args.log_path}")
    if args.doc_id is not None:
        out_lines.append(f"[info] filter: doc_id={args.doc_id}")
    out_lines.append("")

    emitted = 0
    for (doc_id, idx, dump_seed, packedN, H, block_text) in section_ranges:
        if args.doc_id is not None and doc_id != args.doc_id:
            continue

        rows_raw = extract_nn_rows(block_text)
        if not rows_raw:
            continue

        rows: List[PairRow] = []
        for (u_id, u_x, v_id, v_x, d1) in rows_raw:
            if 0 <= u_id < vocab.shape[0] and 0 <= v_id < vocab.shape[0]:
                l2 = l2dist(vocab, u_id, v_id)
            else:
                l2 = float("nan")
            rows.append(PairRow(u_id=u_id, u_x=u_x, v_id=v_id, v_x=v_x, d1=d1, l2=l2))

        out_lines.append(f"=== doc_id={doc_id} idx={idx} dump_seed={dump_seed} packedN={packedN} H={H} ===")
        out_lines.append("")
        out_lines.append("[table] log-order (+ -> - NN) with L2")
        out_lines.append(format_table(rows, top=top))
        out_lines.append("")

        rows_sorted = sorted(rows, key=lambda r: (r.d1, r.l2))
        out_lines.append("[table] d1-sorted (then L2) with L2")
        out_lines.append(format_table(rows_sorted, top=top))
        out_lines.append("")

        emitted += 1

    if emitted == 0:
        out_lines.append("[warn] no matching doc sections / no NN rows found.")

    out_text = "\n".join(out_lines)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"[written] {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
