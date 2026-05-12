#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SO_DIR = "/mnt/c/Users/成見/0905_work/native/build"
DEFAULT_DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"


def to_measure(obj):
    """
    queries.npy / dataset.npy の1要素を list[(vid, weight)] に変換する。
    語彙ID列なら一様重みを付ける。
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


def parse_doc_range(spec: str, n_docs: int):
    """
    例:
      all       -> 全doc
      0:100     -> 0..99
      0:100:5   -> 0,5,10,...,95
      1,4,10    -> 指定doc
    """
    spec = spec.strip()
    if spec == "all":
        return np.arange(n_docs, dtype=np.int32)
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            a, b = parts
            step = 1
        elif len(parts) == 3:
            a, b, step = parts
        else:
            raise ValueError("--docs は all, A:B, A:B:STEP, または comma list で指定してください")
        if step <= 0:
            raise ValueError("step must be positive")
        ids = list(range(a, min(b, n_docs), step))
        return np.array(ids, dtype=np.int32)
    return np.array([int(x) for x in spec.split(",") if x.strip()], dtype=np.int32)


def sort_topk(ids, scores):
    order = np.argsort(scores)
    return ids[order], scores[order]


def overlap_at_k(gt_ids, pred_ids, k=None):
    if k is None:
        k = len(gt_ids)
    gt = set(map(int, gt_ids[:k]))
    pr = set(map(int, pred_ids[:k]))
    if not gt:
        return float("nan")
    return len(gt & pr) / len(gt)


def exact_emd_topk(vocab, query_measure, dataset_measures, input_ids, topk):
    """
    Exact EMD top-k をPOTで外部計算する。
    距離行列は rows=query, cols=doc とし、ot.emd2(q_w, d_w, M) で計算する。
    """
    try:
        import ot
    except ImportError as e:
        raise RuntimeError("POT が見つかりません。`pip install POT` を実行してください。") from e

    q_ids = np.array([v for v, _ in query_measure], dtype=np.int64)
    q_w = np.array([w for _, w in query_measure], dtype=np.float64)
    q_vecs = vocab[q_ids].astype(np.float64)

    distances = []
    for doc_id in input_ids:
        doc = dataset_measures[int(doc_id)]
        d_ids = np.array([v for v, _ in doc], dtype=np.int64)
        d_w = np.array([w for _, w in doc], dtype=np.float64)
        d_vecs = vocab[d_ids].astype(np.float64)

        diff = q_vecs[:, None, :] - d_vecs[None, :, :]
        M = np.linalg.norm(diff, axis=2).astype(np.float64)
        score = float(ot.emd2(q_w, d_w, M))
        distances.append((score, int(doc_id)))

    distances.sort(key=lambda x: x[0])
    top = distances[:topk]
    ids = np.array([doc_id for _, doc_id in top], dtype=np.int32)
    scores = np.array([score for score, _ in top], dtype=np.float32)
    return ids, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--so_dir", default=DEFAULT_SO_DIR)
    parser.add_argument("--qid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--docs", default="all", help="all, 0:1000, 0:1000:5, or comma list")
    parser.add_argument("--skip_exact", action="store_true", help="Exact EMD top-kを省略する")
    parser.add_argument("--verbose_cpp", action="store_true", help="C++側debugを表示する")
    args = parser.parse_args()

    sys.path.insert(0, args.so_dir)
    import ot_estimators_twotree as ote2

    data_dir = Path(args.data_dir)
    vocab = np.load(data_dir / "vocab.npy").astype(np.float32)
    queries = np.load(data_dir / "queries.npy", allow_pickle=True)
    dataset = np.load(data_dir / "dataset.npy", allow_pickle=True)

    query = to_measure(queries[args.qid])
    dataset_measures = [to_measure(x) for x in dataset]
    input_ids = parse_doc_range(args.docs, len(dataset_measures))

    if args.topk > len(input_ids):
        raise ValueError(f"topk={args.topk} is larger than number of candidate docs={len(input_ids)}")

    print("[data]")
    print(f"  data_dir={args.data_dir}")
    print(f"  vocab.shape={vocab.shape} dtype={vocab.dtype}")
    print(f"  qid={args.qid}")
    print(f"  |query|={len(query)} sum(query)={sum(w for _, w in query):.12f}")
    print(f"  docs={args.docs} count={len(input_ids)}")
    print(f"  seed={args.seed} topk={args.topk}")
    print()

    # C++ solver
    solver = ote2.OTEstimators()
    try:
        solver.load_vocabulary(vocab, args.seed, args.verbose_cpp)
    except TypeError:
        solver.load_vocabulary(vocab, args.seed)
    solver.load_dataset(dataset_measures)

    root_parts = solver.get_root_parts() if hasattr(solver, "get_root_parts") else -1
    root_parts_second = solver.get_root_parts_second() if hasattr(solver, "get_root_parts_second") else -1
    print("[tree root parts]")
    print(f"  root_parts        = {root_parts}")
    print(f"  root_parts_second = {root_parts_second}")
    print()

    # Flowtree top-k
    output_ids_flow = np.zeros(args.topk, dtype=np.int32)
    output_scores_flow = np.zeros(args.topk, dtype=np.float32)
    t0 = time.perf_counter()
    solver.flowtree_rank(query, input_ids, output_ids_flow, output_scores_flow, True)
    time_flow = time.perf_counter() - t0
    output_ids_flow, output_scores_flow = sort_topk(output_ids_flow, output_scores_flow)

    # Quadtree top-k
    output_ids_quad = np.zeros(args.topk, dtype=np.int32)
    output_scores_quad = np.zeros(args.topk, dtype=np.float32)
    t0 = time.perf_counter()
    solver.quadtree_rank(query, input_ids, output_ids_quad, output_scores_quad, True)
    time_quad = time.perf_counter() - t0
    output_ids_quad, output_scores_quad = sort_topk(output_ids_quad, output_scores_quad)

    # Exact top-k
    exact_ids = None
    exact_scores = None
    time_exact = None
    if not args.skip_exact:
        t0 = time.perf_counter()
        exact_ids, exact_scores = exact_emd_topk(vocab, query, dataset_measures, input_ids, args.topk)
        time_exact = time.perf_counter() - t0

    print("=== Flowtree ===")
    print("ids   :", output_ids_flow)
    print("scores:", output_scores_flow)
    print(f"time  : {time_flow:.6f}s")
    print()

    print("=== Quadtree ===")
    print("ids   :", output_ids_quad)
    print("scores:", output_scores_quad)
    print(f"time  : {time_quad:.6f}s")
    print()

    if exact_ids is not None:
        print("=== Exact EMD ===")
        print("ids   :", exact_ids)
        print("scores:", exact_scores)
        print(f"time  : {time_exact:.6f}s")
        print()

        print("=== Accuracy vs Exact top-k ===")
        print(f"Flowtree Overlap@{args.topk}: {overlap_at_k(exact_ids, output_ids_flow):.3f}")
        print(f"Quadtree Overlap@{args.topk}: {overlap_at_k(exact_ids, output_ids_quad):.3f}")
        print()

        print("=== Exact scores for predicted docs ===")
        # Flowtree/Quadtreeで選ばれたdocのexact scoreも確認する
        pred_union = sorted(set(map(int, output_ids_flow)) | set(map(int, output_ids_quad)))
        exact_map_ids, exact_map_scores = exact_emd_topk(
            vocab,
            query,
            dataset_measures,
            np.array(pred_union, dtype=np.int32),
            len(pred_union),
        )
        exact_score_map = {int(i): float(s) for i, s in zip(exact_map_ids, exact_map_scores)}
        print("doc_id,flowtree_score,quadtree_score,exact_score")
        flow_score_map = {int(i): float(s) for i, s in zip(output_ids_flow, output_scores_flow)}
        quad_score_map = {int(i): float(s) for i, s in zip(output_ids_quad, output_scores_quad)}
        for doc_id in pred_union:
            print(
                f"{doc_id},"
                f"{flow_score_map.get(doc_id, np.nan):.9f},"
                f"{quad_score_map.get(doc_id, np.nan):.9f},"
                f"{exact_score_map.get(doc_id, np.nan):.9f}"
            )


if __name__ == "__main__":
    main()
