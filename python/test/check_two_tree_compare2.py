import argparse
import sys
import numpy as np

DEFAULT_SO_DIR = "/mnt/c/Users/成見/0905_work/native/build"
DEFAULT_DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"


def to_measure(obj):
    """
    queries.npy / dataset.npy の1要素を list[(vid, weight)] に変換する。
    今の実データは語彙ID列なので、一様重みを付ける。
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        elif obj.ndim == 1:
            ids = [int(x) for x in obj.tolist()]
            if len(ids) == 0:
                return []
            w = 1.0 / len(ids)
            return [(v, w) for v in ids]

    if isinstance(obj, list):
        if len(obj) == 0:
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


def compare_nn(nn1, nn2, limit=40):
    """
    nn1, nn2: list of tuples (u_id, u_x, v_id, v_x, d1)
    u_id ごとに比較する。
    """
    d1 = {row[0]: row for row in nn1}
    d2 = {row[0]: row for row in nn2}

    common = sorted(set(d1.keys()) & set(d2.keys()))
    only1 = sorted(set(d1.keys()) - set(d2.keys()))
    only2 = sorted(set(d2.keys()) - set(d1.keys()))

    same_v = 0
    diff_v = 0
    same_d1 = 0
    diff_d1 = 0
    rows_diff = []

    for u in common:
        r1 = d1[u]
        r2 = d2[u]

        if r1[2] == r2[2]:
            same_v += 1
        else:
            diff_v += 1

        if r1[4] == r2[4]:
            same_d1 += 1
        else:
            diff_d1 += 1

        if r1[2] != r2[2] or r1[4] != r2[4]:
            rows_diff.append((u, r1, r2))

    print("[nn_1d compare]")
    print("  common_u =", len(common))
    print("  only_tree1_u =", len(only1))
    print("  only_tree2_u =", len(only2))
    print("  same picked v =", same_v)
    print("  diff picked v =", diff_v)
    print("  same d1 =", same_d1)
    print("  diff d1 =", diff_d1)
    print()

    print(f"[nn_1d diff head] (show up to {limit})")
    for i, (u, r1, r2) in enumerate(rows_diff[:limit], start=1):
        print(
            f"{i:3d}  u={u:6d} | "
            f"tree1: (u_x={r1[1]:3d}, v={r1[2]:6d}, v_x={r1[3]:3d}, d1={r1[4]:3d}) | "
            f"tree2: (u_x={r2[1]:3d}, v={r2[2]:6d}, v_x={r2[3]:3d}, d1={r2[4]:3d})"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--so_dir", default=DEFAULT_SO_DIR)
    parser.add_argument("--qid", type=int, default=1)
    parser.add_argument("--doc_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1275)
    parser.add_argument("--verbose_cpp", action="store_true", default=False,
                        help="C++側のCHECK_SPLIT/packed subtree/dot出力を有効にする。")
    parser.add_argument("--no_verbose_cpp", dest="verbose_cpp", action="store_false",
                        help="C++側の詳細debugを止める。")
    parser.add_argument("--head", type=int, default=40)
    args = parser.parse_args()

    sys.path.insert(0, args.so_dir)
    import ot_estimators_twotree as ote2

    vocab = np.load(f"{args.data_dir}/vocab.npy").astype(np.float32)
    queries = np.load(f"{args.data_dir}/queries.npy", allow_pickle=True)
    dataset = np.load(f"{args.data_dir}/dataset.npy", allow_pickle=True)

    query = to_measure(queries[args.qid])
    doc = to_measure(dataset[args.doc_id])

    print("[data]")
    print("  data_dir =", args.data_dir)
    print("  so_dir   =", args.so_dir)
    print("  vocab.shape =", vocab.shape, "dtype =", vocab.dtype)
    print("  qid =", args.qid, "doc_id =", args.doc_id)
    print("  seed =", args.seed, "verbose_cpp =", args.verbose_cpp)
    print("  |query| =", len(query), "sum(query) =", sum(w for _, w in query))
    print("  |doc|   =", len(doc), "sum(doc)   =", sum(w for _, w in doc))
    print()

    solver = ote2.OTEstimators()

    # 新C++版: load_vocabulary(vocab, seed, verbose)
    # 古いsoを誤って読んだ場合にも落ちないようにfallbackを置く。
    try:
        solver.load_vocabulary(vocab, args.seed, args.verbose_cpp)
    except TypeError:
        print("[warn] load_vocabulary(vocab, seed, verbose) が使えません。古い .so を読んでいる可能性があります。")
        try:
            solver.load_vocabulary(vocab, args.seed)
        except TypeError:
            solver.load_vocabulary(vocab)

    solver.load_dataset([doc])

    if hasattr(solver, "get_root_parts"):
        print("[tree root parts from getter]")
        print("  root_parts        =", solver.get_root_parts())
        print("  root_parts_second =", solver.get_root_parts_second())
        print()

    # flowtree_query を内部で動かすため、flowtree_rank を1件だけ呼ぶ。
    input_ids = np.array([0], dtype=np.int32)
    output_ids = np.zeros(1, dtype=np.int32)
    output_scores = np.zeros(1, dtype=np.float32)
    solver.flowtree_rank(query, input_ids, output_ids, output_scores, True)

    print("[flowtree_rank pair result]")
    print("  output_ids    =", output_ids)
    print("  output_scores =", output_scores)
    print()

    # tree1
    packed_x1 = solver.get_last_packed_x()
    plus_pts1 = solver.get_last_plus_pts()
    minus_pts1 = solver.get_last_minus_pts()
    nn1 = solver.get_last_nn_1d()

    # tree2
    packed_x2 = solver.get_last_packed_x_second()
    plus_pts2 = solver.get_last_plus_pts_second()
    minus_pts2 = solver.get_last_minus_pts_second()
    nn2 = solver.get_last_nn_1d_second()

    h = args.head

    print("[tree1 summary]")
    print("  len(packed_x1)  =", len(packed_x1))
    print("  len(plus_pts1)  =", len(plus_pts1))
    print("  len(minus_pts1) =", len(minus_pts1))
    print("  len(nn1)        =", len(nn1))
    print()

    print("[tree2 summary]")
    print("  len(packed_x2)  =", len(packed_x2))
    print("  len(plus_pts2)  =", len(plus_pts2))
    print("  len(minus_pts2) =", len(minus_pts2))
    print("  len(nn2)        =", len(nn2))
    print()

    print("[packed_x head]")
    print("  tree1:", packed_x1[:h])
    print("  tree2:", packed_x2[:h])
    print()

    print("[plus_pts head] (vid, x, mass)")
    print("  tree1:")
    for row in plus_pts1[:h]:
        print("   ", row)
    print("  tree2:")
    for row in plus_pts2[:h]:
        print("   ", row)
    print()

    print("[minus_pts head] (vid, x, mass)")
    print("  tree1:")
    for row in minus_pts1[:h]:
        print("   ", row)
    print("  tree2:")
    for row in minus_pts2[:h]:
        print("   ", row)
    print()

    print("[nn_1d head] (u_id, u_x, v_id, v_x, d1)")
    print("  tree1:")
    for row in nn1[:h]:
        print("   ", row)
    print("  tree2:")
    for row in nn2[:h]:
        print("   ", row)
    print()

    compare_nn(nn1, nn2, limit=h)

    if args.verbose_cpp:
        print("[dot hint]")
        print(f"  cd /mnt/c/Users/成見/0905_work/python/test/")
        print(f"  dot -Tsvg tree1_seed{args.seed}.dot -o tree1_seed{args.seed}.svg")
        print(f"  dot -Tsvg tree2_seed{args.seed}.dot -o tree2_seed{args.seed}.svg")


if __name__ == "__main__":
    main()
