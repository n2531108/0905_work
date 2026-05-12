#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_batch_csv.py

Batch CSV（nn_two_logs_pick_greedy_batch_0313.py 系）を読み取り、集計して表示する。

特徴:
- pandas 不要（標準ライブラリのみ）
- --seed A/B/both で出力対象を切替（デフォルト A）
- --mode minimal/full で出力を絞れる（デフォルト minimal）
- 旧CSV/新CSVの列名差分に自動対応（baseline_A vs costA_base など）

Usage:
  python3 summarize_batch_csv.py --csv /path/to/batch.csv --topk 20
  python3 summarize_batch_csv.py --csv /path/to/batch.csv --seed B --mode minimal
  python3 summarize_batch_csv.py --csv /path/to/batch.csv --seed both --mode full
"""

import argparse
import csv
import math
import statistics
from typing import Dict, List, Any, Tuple


# ----------------------------
# Helpers: safe parsing
# ----------------------------

def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _to_int(x: Any) -> int:
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        # some fields may be floats like "33.0"
        return int(float(s))
    except Exception:
        return 0


def _is_finite(x: float) -> bool:
    return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))


def _get_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] != "":
            return d[k]
    return default


# ----------------------------
# Column mapping (old/new)
# ----------------------------

def _col(row: Dict[str, Any], name: str) -> Any:
    """Return value for a canonical name, mapping old/new columns."""
    mapping = {
        # identifiers
        "qid": ["qid"],
        "doc_id": ["doc_id"],
        "status": ["status"],

        # prematch summary
        "prematch_ratio": ["prematch_ratio", "prem_ratio"],
        "prematched_flow": ["prematched_flow", "prem_flow"],
        "prematch_cost": ["prematch_cost", "prem_cost"],
        "delta_total": ["delta_total", "delta"],
        "residual_mass": ["residual_mass", "rem_plus_sum"],  # old style used rem_plus_sum==rem_minus_sum
        "nz_plus": ["nz_plus"],
        "nz_minus": ["nz_minus"],

        # seedA costs
        "baseline_A": ["baseline_A", "costA_base"],
        "residual_A": ["residual_A", "costA_res"],
        "total_A": ["total_A", "costA_total"],
        "improve_A": ["improve_A"],

        # seedB costs
        "baseline_B": ["baseline_B", "costB_base"],
        "residual_B": ["residual_B", "costB_res"],
        "total_B": ["total_B", "costB_total"],
        "improve_B": ["improve_B"],

        # timings (optional)
        "t_total": ["t_total"],
    }
    return _get_first(row, mapping.get(name, [name]), default=None)


def _compute_improve_if_missing(row: Dict[str, Any], seed: str) -> float:
    """
    improve = baseline - total
    If improve column exists, use it; else compute if possible.
    """
    if seed == "A":
        imp = _to_float(_col(row, "improve_A"))
        if _is_finite(imp):
            return imp
        base = _to_float(_col(row, "baseline_A"))
        total = _to_float(_col(row, "total_A"))
        if _is_finite(base) and _is_finite(total):
            return base - total
        return float("nan")
    else:
        imp = _to_float(_col(row, "improve_B"))
        if _is_finite(imp):
            return imp
        base = _to_float(_col(row, "baseline_B"))
        total = _to_float(_col(row, "total_B"))
        if _is_finite(base) and _is_finite(total):
            return base - total
        return float("nan")


def _compute_rel_improve(base: float, improve: float) -> float:
    """
    relative improvement = (baseline - total) / baseline = improve / baseline
    """
    if not _is_finite(base) or not _is_finite(improve) or base <= 0:
        return float("nan")
    return improve / base


# ----------------------------
# Stats
# ----------------------------

def _quantile(xs: List[float], q: float) -> float:
    xs = [x for x in xs if _is_finite(x)]
    if not xs:
        return float("nan")
    xs.sort()
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _basic_stats(xs: List[float]) -> Dict[str, float]:
    xs_f = [x for x in xs if _is_finite(x)]
    if not xs_f:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
        }
    mean = statistics.fmean(xs_f)
    median = statistics.median(xs_f)
    std = statistics.pstdev(xs_f) if len(xs_f) >= 2 else 0.0
    return {
        "n": len(xs_f),
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(xs_f),
        "p25": _quantile(xs_f, 0.25),
        "p75": _quantile(xs_f, 0.75),
        "max": max(xs_f),
    }


def _corr(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if _is_finite(x) and _is_finite(y)]
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = sum((x - mx) ** 2 for x in xvals)
    deny = sum((y - my) ** 2 for y in yvals)
    if denx <= 0 or deny <= 0:
        return float("nan")
    return num / math.sqrt(denx * deny)


# ----------------------------
# Printing
# ----------------------------

def _print_block_header(title: str):
    print()
    print(f"=== {title} ===")


def _fmt(x: float, nd: int = 6) -> str:
    if not _is_finite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _print_stats(prefix: str, st: Dict[str, float]):
    print(f"  {prefix}_n       : {st['n']}")
    print(f"  {prefix}_mean    : {_fmt(st['mean'])}")
    print(f"  {prefix}_median  : {_fmt(st['median'])}")
    print(f"  {prefix}_std     : {_fmt(st['std'])}")
    print(f"  {prefix}_min     : {_fmt(st['min'])}")
    print(f"  {prefix}_p25     : {_fmt(st['p25'])}")
    print(f"  {prefix}_p75     : {_fmt(st['p75'])}")
    print(f"  {prefix}_max     : {_fmt(st['max'])}")


def _print_top_table(rows: List[Dict[str, Any]], seed: str, topk: int, worst: bool = False):
    def get_imp(r):
        return r["improve"]

    rows2 = [r for r in rows if _is_finite(r["improve"])]
    rows2.sort(key=get_imp, reverse=not worst)
    rows2 = rows2[:topk]

    title = f"{'WORST' if worst else 'TOP'} {topk} by improve_{seed}"
    _print_block_header(title)

    print(" doc_id  baseline   total     improve   rel_impr   prem_ratio  prem_flow")
    for r in rows2:
        print(
            f" {r['doc_id']:5d}"
            f"  {_fmt(r['baseline']):>8s}"
            f"  {_fmt(r['total']):>8s}"
            f"  {_fmt(r['improve']):>8s}"
            f"  {_fmt(r['rel_improve']):>8s}"
            f"  {_fmt(r['prematch_ratio']):>9s}"
            f"  {_fmt(r['prematched_flow']):>9s}"
        )


# ----------------------------
# Core: build parsed view for a seed
# ----------------------------

def _build_parsed(rows_raw: List[Dict[str, Any]], seed: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows_ok = 0
    rows_ng = 0
    parsed: List[Dict[str, Any]] = []

    for row in rows_raw:
        status = str(_col(row, "status") or "ok").strip()
        if status == "" or status.lower() == "ok":
            rows_ok += 1
        else:
            rows_ng += 1

        doc_id = _to_int(_col(row, "doc_id"))
        prem_ratio = _to_float(_col(row, "prematch_ratio"))
        prem_flow = _to_float(_col(row, "prematched_flow"))
        prem_cost = _to_float(_col(row, "prematch_cost"))

        if seed == "A":
            base = _to_float(_col(row, "baseline_A"))
            total = _to_float(_col(row, "total_A"))
            improve = _compute_improve_if_missing(row, "A")
        else:
            base = _to_float(_col(row, "baseline_B"))
            total = _to_float(_col(row, "total_B"))
            improve = _compute_improve_if_missing(row, "B")

        rel_improve = _compute_rel_improve(base, improve)

        parsed.append({
            "doc_id": doc_id,
            "status": status,
            "baseline": base,
            "total": total,
            "improve": improve,
            "rel_improve": rel_improve,
            "prematch_ratio": prem_ratio,
            "prematched_flow": prem_flow,
            "prematch_cost": prem_cost,
        })

    return parsed, {"rows_ok": rows_ok, "rows_ng": rows_ng}


def _print_seed_section(parsed: List[Dict[str, Any]], seed: str, topk: int, mode: str):
    # Improvement stats
    improves = [r["improve"] for r in parsed]
    rel_improves = [r["rel_improve"] for r in parsed]
    st_imp = _basic_stats(improves)
    st_rel = _basic_stats(rel_improves)

    pos = [x for x in improves if _is_finite(x) and x > 0]
    neg = [x for x in improves if _is_finite(x) and x < 0]
    p_pos = (len(pos) / st_imp["n"]) if st_imp["n"] > 0 else float("nan")
    p_neg = (len(neg) / st_imp["n"]) if st_imp["n"] > 0 else float("nan")

    _print_block_header(f"improvement stats seed{seed}")
    _print_stats(f"improve_{seed}", st_imp)
    print(f"  p_improve_pos_{seed} : {_fmt(p_pos)}")
    print(f"  p_improve_neg_{seed} : {_fmt(p_neg)}")
    _print_stats(f"rel_improve_{seed}", st_rel)
    rel_pos = [x for x in rel_improves if _is_finite(x) and x > 0]
    p_rel_pos = (len(rel_pos) / st_rel["n"]) if st_rel["n"] > 0 else float("nan")
    print(f"  p_rel_improve_pos_{seed} : {_fmt(p_rel_pos)}")

    # Prematch stats
    prem_ratios = [r["prematch_ratio"] for r in parsed]
    prem_flows = [r["prematched_flow"] for r in parsed]
    st_pr = _basic_stats(prem_ratios)
    st_pf = _basic_stats(prem_flows)

    _print_block_header(f"prematch stats (shared) seed{seed}")
    _print_stats("prematch_ratio", st_pr)
    _print_stats("prematched_flow", st_pf)

    # Top/Worst tables
    _print_top_table(parsed, seed, topk, worst=False)
    _print_top_table(parsed, seed, topk, worst=True)

    if mode != "full":
        return

    # Correlations
    _print_block_header(f"correlations seed{seed}")
    prem_costs = [r["prematch_cost"] for r in parsed]
    print(f"  corr(improve_{seed}, prematch_ratio)      = {_fmt(_corr(improves, prem_ratios), 4)}")
    print(f"  corr(rel_improve_{seed}, prematch_ratio)  = {_fmt(_corr(rel_improves, prem_ratios), 4)}")
    print(f"  corr(improve_{seed}, prematch_cost)       = {_fmt(_corr(improves, prem_costs), 4)}")
    print(f"  corr(rel_improve_{seed}, prematch_cost)   = {_fmt(_corr(rel_improves, prem_costs), 4)}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to batch csv")
    ap.add_argument("--topk", type=int, default=20, help="top-k docs to show")
    ap.add_argument("--seed", choices=["A", "B", "both"], default="A",
                    help="which seed side to focus on (default: A). use 'both' for A+B")
    ap.add_argument("--mode", choices=["minimal", "full"], default="minimal",
                    help="output verbosity: minimal or full (default: minimal)")
    args = ap.parse_args()

    # Load CSV
    rows_raw: List[Dict[str, Any]] = []
    with open(args.csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_raw.append(row)

    rows_total = len(rows_raw)

    # Dataset header
    _print_block_header("dataset")
    print(f"  rows_total        : {rows_total}")
    print(f"  focus_seed        : {args.seed}")
    print(f"  mode              : {args.mode}")

    # Decide which seeds to print
    seeds_to_print = ["A", "B"] if args.seed == "both" else [args.seed]

    # Build and print sections
    # Note: rows_ok/ng may differ if status differs per row, but usually same.
    for s in seeds_to_print:
        parsed, c = _build_parsed(rows_raw, s)

        # print ok/ng per section (useful if any row has NG)
        _print_block_header(f"dataset status (seed{s})")
        print(f"  rows_ok           : {c['rows_ok']}")
        print(f"  rows_ng           : {c['rows_ng']}")

        _print_seed_section(parsed, s, args.topk, args.mode)


if __name__ == "__main__":
    main()