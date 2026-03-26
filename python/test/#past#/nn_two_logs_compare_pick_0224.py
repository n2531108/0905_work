#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn_two_logs_compare_pick_0224.py

目的:
- 同じ query-doc ペアに対する 2本の FlowTree 1D-NN ログ（seed違い）を読み込む
- 各 u (query側の + 葉語彙) について、木A/Bそれぞれの最近傍候補 v と d1 を取得
- vocab.npy から元空間ベクトルを引いて L2(u,vA), L2(u,vB) を計算
- 「一方の木（既定A）の d1 小さい順」に並べ、
  各 u で L2 の小さい方を採用する表を出力する

現時点では「比較＋採用表の出力」まで。
（greedyプレマッチ本体は別段階）

使い方例:
python3 nn_two_logs_compare_pick_0224.py \
  --data_dir "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full" \
  --logA "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed110.log" \
  --logB "/mnt/c/Users/成見/0905_work/tmp_real_dump/run_seed111.log" \
  --doc_id 0 \
  --priority A \
  --out "/mnt/c/Users/成見/0905_work/tmp_real_dump/compare_pick_seed110_vs_111_doc0.txt"

ログの想定フォーマット（あなたの出力）:
=== doc_id=0 idx=0 dump_seed=111 packedN=159 H=4 ===
...
=== + -> -  scan NN ===
 u_id  u_x   ->   v_id  v_x   d1
    50    0           246    2    2
...
[check +->-] OK
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# ------------------------------
# データ構造
# ------------------------------

@dataclass
class NNRow:
    u_id: int
    u_x: int
    v_id: int
    v_x: int
    d1: int


@dataclass
class CompareRow:
    u_id: int
    # A
    u_x_A: int
    v_id_A: int
    v_x_A: int
    d1_A: int
    l2_A: float
    # B
    u_x_B: int
    v_id_B: int
    v_x_B: int
    d1_B: int
    l2_B: float
    # pick
    pick: str          # "A" or "B"
    picked_v: int
    picked_d1: int
    picked_l2: float
    reason: str


# ------------------------------
# ユーティリティ
# ------------------------------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def l2_distance(vocab: np.ndarray, i: int, j: int) -> float:
    # vocab: (V, D) float32
    a = vocab[i]
    b = vocab[j]
    d = a - b
    # floatで返す
    return float(np.sqrt(np.dot(d, d)))


# ------------------------------
# ログ解析
# ------------------------------

_DOC_HEADER_RE = re.compile(
    r"^===\s*doc_id=(?P<docid>\d+)\s+idx=(?P<idx>\d+)\s+dump_seed=(?P<seed>-?\d+)\s+packedN=(?P<packed>\d+)\s+H=(?P<H>\d+)\s*===\s*$",
    re.M,
)

# 表本体の1行: u_id u_x v_id v_x d1
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.M
)


def extract_doc_section(log_text: str, doc_id: int) -> str:
    """
    指定 doc_id のセクションを抽出。
    次の '=== doc_id=...' までを1セクションとみなす。
    """
    matches = list(_DOC_HEADER_RE.finditer(log_text))
    if not matches:
        raise ValueError("doc header not found in log")

    target_idx = None
    for k, m in enumerate(matches):
        if int(m.group("docid")) == doc_id:
            target_idx = k
            break

    if target_idx is None:
        doc_ids = [int(m.group("docid")) for m in matches]
        raise ValueError(f"doc_id={doc_id} not found in log. available={doc_ids}")

    start = matches[target_idx].start()
    end = matches[target_idx + 1].start() if target_idx + 1 < len(matches) else len(log_text)
    return log_text[start:end]


def parse_nn_table_from_doc_section(section_text: str) -> Dict[int, NNRow]:
    """
    doc section 内の '=== + -> -  scan NN ===' テーブルを parse して
    u_id -> NNRow を返す
    """
    marker = "=== + -> -  scan NN ==="
    p = section_text.find(marker)
    if p < 0:
        raise ValueError("NN table marker not found: '=== + -> -  scan NN ==='")

    tail = section_text[p + len(marker):]

    rows = _ROW_RE.findall(tail)
    if not rows:
        raise ValueError("No NN rows found after NN table marker")

    out: Dict[int, NNRow] = {}
    for u_id, u_x, v_id, v_x, d1 in rows:
        r = NNRow(
            u_id=int(u_id),
            u_x=int(u_x),
            v_id=int(v_id),
            v_x=int(v_x),
            d1=int(d1),
        )
        # u_id重複は後勝ち（通常は起きない）
        out[r.u_id] = r
    return out


def parse_doc_meta(section_text: str) -> Dict[str, int]:
    m = _DOC_HEADER_RE.search(section_text)
    if not m:
        return {}
    return {
        "doc_id": int(m.group("docid")),
        "idx": int(m.group("idx")),
        "dump_seed": int(m.group("seed")),
        "packedN": int(m.group("packed")),
        "H": int(m.group("H")),
    }


# ------------------------------
# 比較・採用
# ------------------------------

def choose_by_l2(
    a: NNRow,
    b: NNRow,
    l2_a: float,
    l2_b: float,
    eps: float = 1e-12,
) -> Tuple[str, int, int, float, str]:
    """
    L2 が小さい方を採用。
    同値時のタイブレーク:
      1) d1 が小さい方
      2) それでも同値なら A 優先
    """
    if l2_a + eps < l2_b:
        return "A", a.v_id, a.d1, l2_a, "L2_A < L2_B"
    if l2_b + eps < l2_a:
        return "B", b.v_id, b.d1, l2_b, "L2_B < L2_A"

    # L2 tie -> d1 smaller
    if a.d1 < b.d1:
        return "A", a.v_id, a.d1, l2_a, "L2 tie, d1_A < d1_B"
    if b.d1 < a.d1:
        return "B", b.v_id, b.d1, l2_b, "L2 tie, d1_B < d1_A"

    # still tie -> A
    return "A", a.v_id, a.d1, l2_a, "L2 tie, d1 tie -> pick A"


def build_compare_rows(
    vocab: np.ndarray,
    mapA: Dict[int, NNRow],
    mapB: Dict[int, NNRow],
    priority: str = "A",
    require_common: bool = True,
) -> Tuple[List[CompareRow], Dict[str, List[int]]]:
    """
    priority:
      "A" -> Aのd1昇順（u基準）
      "B" -> Bのd1昇順
    """
    setA = set(mapA.keys())
    setB = set(mapB.keys())
    common = sorted(setA & setB)
    onlyA = sorted(setA - setB)
    onlyB = sorted(setB - setA)

    if require_common and (onlyA or onlyB):
        # 厳密に合わせたい場合はここで止める
        raise ValueError(
            f"u set mismatch: onlyA={len(onlyA)} onlyB={len(onlyB)} "
            f"(use --allow_mismatch to proceed on common only)"
        )

    rows: List[CompareRow] = []
    for u in common:
        a = mapA[u]
        b = mapB[u]
        l2_a = l2_distance(vocab, u, a.v_id)
        l2_b = l2_distance(vocab, u, b.v_id)

        pick, picked_v, picked_d1, picked_l2, reason = choose_by_l2(a, b, l2_a, l2_b)

        rows.append(
            CompareRow(
                u_id=u,
                u_x_A=a.u_x, v_id_A=a.v_id, v_x_A=a.v_x, d1_A=a.d1, l2_A=l2_a,
                u_x_B=b.u_x, v_id_B=b.v_id, v_x_B=b.v_x, d1_B=b.d1, l2_B=l2_b,
                pick=pick, picked_v=picked_v, picked_d1=picked_d1, picked_l2=picked_l2, reason=reason,
            )
        )

    # 並び順（優先木の d1 小さい順、同値はその木の u_x、さらに u_id）
    p = priority.upper()
    if p == "A":
        rows.sort(key=lambda r: (r.d1_A, r.u_x_A, r.u_id))
    elif p == "B":
        rows.sort(key=lambda r: (r.d1_B, r.u_x_B, r.u_id))
    else:
        raise ValueError("priority must be A or B")

    stats = {
        "common_u": common,
        "onlyA_u": onlyA,
        "onlyB_u": onlyB,
    }
    return rows, stats


# ------------------------------
# 表示
# ------------------------------

def format_compare_table(rows: List[CompareRow], priority: str, show_top: int = 10_000) -> str:
    p = priority.upper()
    lines: List[str] = []
    lines.append(f"[table] u-keyed compare & pick (priority by seed{p} d1 asc)")
    lines.append(
        " ord |  u_id |"
        "  u_x(A)  v_id(A)  v_x(A)  d1(A)      L2(A)  ||"
        "  u_x(B)  v_id(B)  v_x(B)  d1(B)      L2(B)  ||"
        " pick  picked_v  picked_d1   picked_L2  reason"
    )
    lines.append("-" * 190)

    for i, r in enumerate(rows[:show_top], start=1):
        lines.append(
            f"{i:4d} | {r.u_id:5d} |"
            f"{r.u_x_A:8d}{r.v_id_A:9d}{r.v_x_A:8d}{r.d1_A:8d}{r.l2_A:11.6f}  ||"
            f"{r.u_x_B:8d}{r.v_id_B:9d}{r.v_x_B:8d}{r.d1_B:8d}{r.l2_B:11.6f}  ||"
            f" {r.pick:>3s}{r.picked_v:10d}{r.picked_d1:11d}{r.picked_l2:12.6f}  {r.reason}"
        )
    return "\n".join(lines)


def format_summary(rows: List[CompareRow], stats: Dict[str, List[int]], metaA: Dict[str, int], metaB: Dict[str, int], priority: str) -> str:
    cnt_pick_A = sum(1 for r in rows if r.pick == "A")
    cnt_pick_B = sum(1 for r in rows if r.pick == "B")
    cnt_same_vid = sum(1 for r in rows if r.v_id_A == r.v_id_B)
    cnt_diff_vid = len(rows) - cnt_same_vid

    # L2改善量（A基準/B基準っぽい指標）
    # ここでは "pickしたL2" と "反対側L2" の差の要約も見たい
    chosen_better_margin = [abs(r.l2_A - r.l2_B) for r in rows]
    avg_margin = float(sum(chosen_better_margin) / len(chosen_better_margin)) if rows else 0.0

    lines: List[str] = []
    lines.append("[summary]")
    if metaA:
        lines.append(
            f"  logA: doc_id={metaA.get('doc_id')} idx={metaA.get('idx')} "
            f"seed={metaA.get('dump_seed')} packedN={metaA.get('packedN')} H={metaA.get('H')}"
        )
    if metaB:
        lines.append(
            f"  logB: doc_id={metaB.get('doc_id')} idx={metaB.get('idx')} "
            f"seed={metaB.get('dump_seed')} packedN={metaB.get('packedN')} H={metaB.get('H')}"
        )
    lines.append(f"  priority = {priority.upper()} (sort by d1 on that log)")
    lines.append(f"  common_u = {len(rows)}")
    lines.append(f"  onlyA_u  = {len(stats.get('onlyA_u', []))}")
    lines.append(f"  onlyB_u  = {len(stats.get('onlyB_u', []))}")
    lines.append(f"  same_NN(v_id_A==v_id_B) = {cnt_same_vid}")
    lines.append(f"  diff_NN(v_id_A!=v_id_B) = {cnt_diff_vid}")
    lines.append(f"  picked A = {cnt_pick_A}")
    lines.append(f"  picked B = {cnt_pick_B}")
    lines.append(f"  avg |L2_A-L2_B| = {avg_margin:.6f}")

    # 参考: d1小さいがL2では負けている件数
    # priority=Aなら "Aのd1を優先順に使ってるが選択はBになる" を数えると意味がある
    if priority.upper() == "A":
        lines.append(f"  (priority=A) rows where B selected = {cnt_pick_B}")
    else:
        lines.append(f"  (priority=B) rows where A selected = {cnt_pick_A}")

    return "\n".join(lines)


def format_pick_only_table(rows: List[CompareRow], show_top: int = 10_000) -> str:
    """
    今後 greedy に渡しやすいよう、採用結果だけの軽い表も出す
    """
    lines: List[str] = []
    lines.append("[table] picked candidate only (for next greedy step)")
    lines.append(" ord   u_id   pick   picked_v   picked_d1   picked_L2")
    lines.append("-" * 58)
    for i, r in enumerate(rows[:show_top], start=1):
        lines.append(
            f"{i:4d}{r.u_id:7d}{r.pick:>7s}{r.picked_v:11d}{r.picked_d1:12d}{r.picked_l2:12.6f}"
        )
    return "\n".join(lines)


# ------------------------------
# メイン
# ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two FlowTree NN logs and pick smaller-L2 candidate per u")
    ap.add_argument("--data_dir", required=True, help="Directory containing vocab.npy")
    ap.add_argument("--logA", required=True, help="NN log for seed A")
    ap.add_argument("--logB", required=True, help="NN log for seed B")
    ap.add_argument("--doc_id", type=int, required=True, help="doc_id section to parse from logs")
    ap.add_argument("--priority", choices=["A", "B", "a", "b"], default="A",
                    help="Sort order priority by d1 of A or B (default: A)")
    ap.add_argument("--out", default=None, help="Output text file path (optional)")
    ap.add_argument("--show_top", type=int, default=1000, help="Max rows to print in tables")
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="Proceed on common-u only even if logA/logB have different u sets")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # vocab 読み込み
    vocab_path = os.path.join(args.data_dir, "vocab.npy")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.npy not found: {vocab_path}")

    vocab = np.load(vocab_path, mmap_mode="r")
    if not isinstance(vocab, np.ndarray) or vocab.ndim != 2:
        raise ValueError(f"vocab.npy must be 2D ndarray, got shape={getattr(vocab, 'shape', None)}")

    # ログ読み込み
    textA = read_text(args.logA)
    textB = read_text(args.logB)

    secA = extract_doc_section(textA, args.doc_id)
    secB = extract_doc_section(textB, args.doc_id)

    metaA = parse_doc_meta(secA)
    metaB = parse_doc_meta(secB)

    mapA = parse_nn_table_from_doc_section(secA)
    mapB = parse_nn_table_from_doc_section(secB)

    rows, stats = build_compare_rows(
        vocab=vocab,
        mapA=mapA,
        mapB=mapB,
        priority=args.priority,
        require_common=(not args.allow_mismatch),
    )

    # 出力構築
    out_lines: List[str] = []
    out_lines.append(f"[input] data_dir={args.data_dir}")
    out_lines.append(f"[input] logA={args.logA}")
    out_lines.append(f"[input] logB={args.logB}")
    out_lines.append(f"[input] doc_id={args.doc_id}")
    out_lines.append("")
    out_lines.append(format_summary(rows, stats, metaA, metaB, args.priority))
    out_lines.append("")
    out_lines.append(format_compare_table(rows, args.priority, show_top=args.show_top))
    out_lines.append("")
    out_lines.append(format_pick_only_table(rows, show_top=args.show_top))
    out_text = "\n".join(out_lines)

    print(out_text)

    if args.out:
        write_text(args.out, out_text)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()