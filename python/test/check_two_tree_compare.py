import sys
import numpy as np

sys.path.append("/mnt/c/Users/成見/0905_work/native/build")
import ot_estimators_twotree as ote2


def to_measure(obj):
    """
    queries.npy / dataset.npy の1要素を
    list[(vid, weight)] に直す簡易版。
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

    raise TypeError(f"unsupported measure format: {type(obj)}")


def compare_nn(nn1, nn2, limit=20):
    """
    nn1, nn2: list of tuples (u_id, u_x, v_id, v_x, d1)
    u_id ごとに比較する
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
    data_dir = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"

    vocab = np.load(f"{data_dir}/vocab.npy").astype(np.float32)
    queries = np.load(f"{data_dir}/queries.npy", allow_pickle=True)
    dataset = np.load(f"{data_dir}/dataset.npy", allow_pickle=True)

    qid = 1
    doc_id = 1

    query = to_measure(queries[qid])
    doc = to_measure(dataset[doc_id])

    print("[data]")
    print("  vocab.shape =", vocab.shape, "dtype =", vocab.dtype)
    print("  qid =", qid, "doc_id =", doc_id)
    print("  |query| =", len(query), "sum(query) =", sum(w for _, w in query))
    print("  |doc|   =", len(doc), "sum(doc)   =", sum(w for _, w in doc))
    print()

    solver = ote2.OTEstimators()
    solver.load_vocabulary(vocab)
    solver.load_dataset([doc])   # stage=2 にするため最小で1件だけ入れる

    # flowtree_query を内部で動かすため、flowtree_rank を1件だけ呼ぶ
    input_ids = np.array([0], dtype=np.int32)
    output_ids = np.zeros(1, dtype=np.int32)
    output_scores = np.zeros(1, dtype=np.float32)
    solver.flowtree_rank(query, input_ids, output_ids, output_scores, True)

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
    print("  tree1:", packed_x1[:40])
    print("  tree2:", packed_x2[:40])
    print()

    print("[plus_pts head] (vid, x, mass)")
    print("  tree1:")
    for row in plus_pts1[:40]:
        print("   ", row)
    print("  tree2:")
    for row in plus_pts2[:40]:
        print("   ", row)
    print()

    print("[minus_pts head] (vid, x, mass)")
    print("  tree1:")
    for row in minus_pts1[:40]:
        print("   ", row)
    print("  tree2:")
    for row in minus_pts2[:40]:
        print("   ", row)
    print()

    print("[nn_1d head] (u_id, u_x, v_id, v_x, d1)")
    print("  tree1:")
    for row in nn1[:0]:
        print("   ", row)
    print("  tree2:")
    for row in nn2[:0]:
        print("   ", row)
    print()

    compare_nn(nn1, nn2, limit=40)


if __name__ == "__main__":
    main()