import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SO_DIR = "/mnt/c/Users/成見/0905_work/native/build"
DEFAULT_DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"


def to_measure(obj):
    """
    queries.npy / dataset.npy の1要素を list[(vid, weight)] に変換する。
    今のデータは語彙ID列なので、一様重みを付ける。
    すでに (vid, weight) 形式なら重みを正規化して返す。
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        elif obj.ndim == 1:
            ids = [int(x) for x in obj.tolist()]
            if not ids:
                return []
            w = 1.0 / len(ids)
            return [(v, w) for v in ids]

    if isinstance(obj, list):
        if not obj:
            return []
        if isinstance(obj[0], (int, np.integer)):
            ids = [int(x) for x in obj]
            w = 1.0 / len(ids)
            return [(v, w) for v in ids]
        if isinstance(obj[0], tuple) and len(obj[0]) == 2:
            pairs = [(int(v), float(w)) for v, w in obj]
            total = sum(w for _, w in pairs)
            if total <= 0:
                raise ValueError("measure total weight must be positive")
            return [(v, w / total) for v, w in pairs]

    raise TypeError(f"unsupported measure format: {type(obj)}")


def exact_emd_cost(vocab, query_measure, doc_measure):
    """
    POTで正確なEMD/Wasserstein-1コストを計算する。
    距離行列は元空間L2距離。
    """
    try:
        import ot
    except ImportError as e:
        raise RuntimeError(
            "POT が見つかりません。WSLで `pip install POT` を実行してください。"
        ) from e

    q_ids = np.array([v for v, _ in query_measure], dtype=np.int64)
    d_ids = np.array([v for v, _ in doc_measure], dtype=np.int64)
    q_w = np.array([w for _, w in query_measure], dtype=np.float64)
    d_w = np.array([w for _, w in doc_measure], dtype=np.float64)

    q = vocab[q_ids].astype(np.float64)
    d = vocab[d_ids].astype(np.float64)

    # rows=doc, cols=query にして既存研究 algorithms.py と同じ向きにする。
    diff = d[:, None, :] - q[None, :, :]
    dm = np.linalg.norm(diff, axis=2).astype(np.float64)
    return float(ot.lp.emd2(d_w, q_w, dm))


def parse_seeds(spec: str):
    """
    例:
      1000:1300      -> 1000,1001,...,1300
      1000:1300:5    -> 1000,1005,...,1300
      1000,1007,1234 -> 指定seedのみ
    """
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            a, b = parts
            step = 1
        elif len(parts) == 3:
            a, b, step = parts
        else:
            raise ValueError("--seeds は A:B, A:B:STEP, または comma list で指定してください")
        if step <= 0:
            raise ValueError("step must be positive")
        return list(range(a, b + 1, step))
    return [int(x) for x in spec.split(",") if x.strip()]


def stat_line(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return f"{name}: no data"
    return (
        f"{name}: "
        f"n={values.size}, "
        f"mean={values.mean():.9f}, "
        f"std={values.std(ddof=0):.9f}, "
        f"min={values.min():.9f}, "
        f"p25={np.quantile(values, 0.25):.9f}, "
        f"median={np.median(values):.9f}, "
        f"p75={np.quantile(values, 0.75):.9f}, "
        f"max={values.max():.9f}"
    )


def corr_or_nan(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--so_dir", default=DEFAULT_SO_DIR)
    parser.add_argument("--qid", type=int, default=1)
    parser.add_argument("--doc_id", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default="1000:1300",
        help="例: 1000:1300, 1000:1300:5, 1000,1001,1234",
    )
    parser.add_argument("--skip_exact", action="store_true")
    parser.add_argument(
        "--csv_out",
        default="seed_cost_parts_q{qid}_d{doc_id}.csv",
        help="CSV保存先。{qid}, {doc_id} を使えます。空文字なら保存しません。",
    )
    parser.add_argument(
        "--verbose_cpp",
        action="store_true",
        help="C++側のshift/root parts debugを表示する。大量出力になるので通常は不要。",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.so_dir)
    import ot_estimators_twotree as ote2

    data_dir = Path(args.data_dir)
    vocab = np.load(data_dir / "vocab.npy").astype(np.float32)
    queries = np.load(data_dir / "queries.npy", allow_pickle=True)
    dataset = np.load(data_dir / "dataset.npy", allow_pickle=True)

    query = to_measure(queries[args.qid])
    doc = to_measure(dataset[args.doc_id])
    seeds = parse_seeds(args.seeds)

    print("[data]")
    print(f"  data_dir={args.data_dir}")
    print(f"  vocab.shape={vocab.shape} dtype={vocab.dtype}")
    print(f"  qid={args.qid} doc_id={args.doc_id}")
    print(f"  |query|={len(query)} sum(query)={sum(w for _, w in query):.12f}")
    print(f"  |doc|  ={len(doc)} sum(doc)  ={sum(w for _, w in doc):.12f}")
    print(f"  seeds={seeds[0]}..{seeds[-1]} count={len(seeds)}")
    print()

    exact = None
    if not args.skip_exact:
        t0 = time.perf_counter()
        exact = exact_emd_cost(vocab, query, doc)
        print(f"[exact_emd] cost={exact:.9f} time={time.perf_counter() - t0:.6f}s")
        print()

    rows = []
    header = [
        "seed",
        "root_parts",
        "root_parts_second",
        "quadtree_cost",
        "flowtree_cost",
        "exact_emd",
        "flowtree_minus_exact",
        "quadtree_minus_exact",
        "abs_flowtree_minus_exact",
        "abs_quadtree_minus_exact",
        "time_sec",
    ]

    print(",".join(header))

    t_all0 = time.perf_counter()
    for seed in seeds:
        t0 = time.perf_counter()

        solver = ote2.OTEstimators()

        # 追加したC++では verbose=false が使える。古いsoを読み込んだ場合のためにfallbackも置く。
        try:
            solver.load_vocabulary(vocab, seed, args.verbose_cpp)
        except TypeError:
            solver.load_vocabulary(vocab, seed)

        solver.load_dataset([doc])

        root_parts = solver.get_root_parts() if hasattr(solver, "get_root_parts") else -1
        root_parts_second = (
            solver.get_root_parts_second() if hasattr(solver, "get_root_parts_second") else -1
        )

        qt_cost = float(solver.quadtree_distance_pair(query, doc))
        ft_cost = float(solver.flowtree_distance_pair(query, doc))
        elapsed = time.perf_counter() - t0

        if exact is None:
            ft_err = np.nan
            qt_err = np.nan
            abs_ft_err = np.nan
            abs_qt_err = np.nan
            exact_print = ""
        else:
            ft_err = ft_cost - exact
            qt_err = qt_cost - exact
            abs_ft_err = abs(ft_err)
            abs_qt_err = abs(qt_err)
            exact_print = f"{exact:.9f}"

        row = {
            "seed": seed,
            "root_parts": root_parts,
            "root_parts_second": root_parts_second,
            "quadtree_cost": qt_cost,
            "flowtree_cost": ft_cost,
            "exact_emd": exact if exact is not None else np.nan,
            "flowtree_minus_exact": ft_err,
            "quadtree_minus_exact": qt_err,
            "abs_flowtree_minus_exact": abs_ft_err,
            "abs_quadtree_minus_exact": abs_qt_err,
            "time_sec": elapsed,
        }
        rows.append(row)

        print(
            f"{seed},"
            f"{root_parts},"
            f"{root_parts_second},"
            f"{qt_cost:.9f},"
            f"{ft_cost:.9f},"
            f"{exact_print},"
            f"{'' if exact is None else f'{ft_err:.9f}'},"
            f"{'' if exact is None else f'{qt_err:.9f}'},"
            f"{'' if exact is None else f'{abs_ft_err:.9f}'},"
            f"{'' if exact is None else f'{abs_qt_err:.9f}'},"
            f"{elapsed:.6f}"
        )

    print()
    print(f"[total_time] {time.perf_counter() - t_all0:.6f}s")
    print()

    # 統計
    root_parts = np.array([r["root_parts"] for r in rows], dtype=np.float64)
    root_parts_second = np.array([r["root_parts_second"] for r in rows], dtype=np.float64)
    qt = np.array([r["quadtree_cost"] for r in rows], dtype=np.float64)
    ft = np.array([r["flowtree_cost"] for r in rows], dtype=np.float64)

    print("[summary]")
    print(stat_line("root_parts", root_parts))
    print(stat_line("root_parts_second", root_parts_second))
    print(stat_line("quadtree_cost", qt))
    print(stat_line("flowtree_cost", ft))

    if exact is not None:
        ft_err = np.array([r["flowtree_minus_exact"] for r in rows], dtype=np.float64)
        qt_err = np.array([r["quadtree_minus_exact"] for r in rows], dtype=np.float64)
        abs_ft_err = np.array([r["abs_flowtree_minus_exact"] for r in rows], dtype=np.float64)
        abs_qt_err = np.array([r["abs_quadtree_minus_exact"] for r in rows], dtype=np.float64)
        print(stat_line("flowtree_minus_exact", ft_err))
        print(stat_line("quadtree_minus_exact", qt_err))
        print(stat_line("abs_flowtree_minus_exact", abs_ft_err))
        print(stat_line("abs_quadtree_minus_exact", abs_qt_err))

        print()
        print("[correlation]")
        print(f"  corr(root_parts, flowtree_cost)           = {corr_or_nan(root_parts, ft):.6f}")
        print(f"  corr(root_parts, abs_flowtree_error)      = {corr_or_nan(root_parts, abs_ft_err):.6f}")
        print(f"  corr(root_parts_second, flowtree_cost)    = {corr_or_nan(root_parts_second, ft):.6f}")
        print(f"  corr(root_parts_second, abs_flowtree_err) = {corr_or_nan(root_parts_second, abs_ft_err):.6f}")

        best = min(rows, key=lambda r: r["abs_flowtree_minus_exact"])
        worst = max(rows, key=lambda r: r["abs_flowtree_minus_exact"])
        print()
        print("[best/worst by abs_flowtree_error]")
        print(
            f"  best : seed={best['seed']} root_parts={best['root_parts']} "
            f"flowtree={best['flowtree_cost']:.9f} abs_err={best['abs_flowtree_minus_exact']:.9f}"
        )
        print(
            f"  worst: seed={worst['seed']} root_parts={worst['root_parts']} "
            f"flowtree={worst['flowtree_cost']:.9f} abs_err={worst['abs_flowtree_minus_exact']:.9f}"
        )

    csv_out = args.csv_out.format(qid=args.qid, doc_id=args.doc_id)
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"[csv] wrote {csv_out}")


if __name__ == "__main__":
    main()
