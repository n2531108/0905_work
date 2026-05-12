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


def main():
    data_dir = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"

    vocab = np.load(f"{data_dir}/vocab.npy").astype(np.float32)
    queries = np.load(f"{data_dir}/queries.npy", allow_pickle=True)
    dataset = np.load(f"{data_dir}/dataset.npy", allow_pickle=True)

    qid = 0
    doc_id = 0

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

    packed_x = solver.get_last_packed_x()
    plus_pts = solver.get_last_plus_pts()
    minus_pts = solver.get_last_minus_pts()
    nn_1d = solver.get_last_nn_1d()

    print("[packed 1D result]")
    print("  len(packed_x) =", len(packed_x))
    print("  len(plus_pts) =", len(plus_pts))
    print("  len(minus_pts) =", len(minus_pts))
    print("  len(nn_1d) =", len(nn_1d))
    print()

    print("[packed_x head]")
    print(packed_x[:30])
    print()

    print("[plus_pts head]  (vid, x, mass)")
    for row in plus_pts[:20]:
        print(" ", row)
    print()

    print("[minus_pts head] (vid, x, mass)")
    for row in minus_pts[:20]:
        print(" ", row)
    print()

    print("[nn_1d head] (u_id, u_x, v_id, v_x, d1)")
    for row in nn_1d[:20]:
        print(" ", row)
    print()

    if len(nn_1d) > 0:
        d1_vals = [row[4] for row in nn_1d]
        print("[nn_1d summary]")
        print("  min d1 =", min(d1_vals))
        print("  max d1 =", max(d1_vals))
        print("  avg d1 =", sum(d1_vals) / len(d1_vals))


if __name__ == "__main__":
    main()