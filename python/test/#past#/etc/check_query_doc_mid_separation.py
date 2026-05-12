import numpy as np


DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"

QID = 1
DOC_ID = 1

SEED = 1234
KS = [6, 12, 24, 48, 100]
DIMS_TO_SHOW = 50


def to_ids(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        elif obj.ndim == 1:
            return np.array([int(x) for x in obj.tolist()], dtype=np.int64)

    if isinstance(obj, list):
        return np.array([int(x) for x in obj], dtype=np.int64)

    raise TypeError(f"unsupported format: {type(obj)}")


def make_python_random_shifts(delta, d, seed=1234):
    """
    注意:
    C++ の std::mt19937_64 とは完全一致しません。
    ただし傾向確認には使えます。
    厳密にC++と合わせたい場合は、C++側で s を出力して読み込んでください。
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, delta, size=d).astype(np.float32)


def summarize_side(label, q_pts, d_pts, mids):
    print(f"\n[{label}]")

    sep_dims = []
    q_all_left_dims = []
    q_all_right_dims = []
    d_all_left_dims = []
    d_all_right_dims = []

    for j in range(q_pts.shape[1]):
        q_vals = q_pts[:, j]
        d_vals = d_pts[:, j]
        mid = mids[j]

        q_left = np.sum(q_vals <= mid)
        q_right = np.sum(q_vals > mid)
        d_left = np.sum(d_vals <= mid)
        d_right = np.sum(d_vals > mid)

        q_side_ratio = max(q_left, q_right) / len(q_vals)
        d_side_ratio = max(d_left, d_right) / len(d_vals)

        q_mean = q_vals.mean()
        d_mean = d_vals.mean()

        # query平均とdoc平均の間にmidがあるか
        mid_between_means = (q_mean < mid < d_mean) or (d_mean < mid < q_mean)

        # かなり強い分離: queryの多数派とdocの多数派が逆側
        q_major_side = "L" if q_left >= q_right else "R"
        d_major_side = "L" if d_left >= d_right else "R"
        majority_opposite = q_major_side != d_major_side

        # ほぼ完全に分離している次元
        strong_opposite = (
            majority_opposite
            and q_side_ratio >= 0.80
            and d_side_ratio >= 0.80
        )

        if strong_opposite:
            sep_dims.append(j)

        if q_left == len(q_vals):
            q_all_left_dims.append(j)
        if q_right == len(q_vals):
            q_all_right_dims.append(j)
        if d_left == len(d_vals):
            d_all_left_dims.append(j)
        if d_right == len(d_vals):
            d_all_right_dims.append(j)

        if j < DIMS_TO_SHOW:
            print(
                f"dim={j:2d} "
                f"mid={mid: .4f} | "
                f"q_mean={q_mean: .4f}, d_mean={d_mean: .4f}, "
                f"mid_between_means={mid_between_means} | "
                f"q L/R={q_left:2d}/{q_right:2d} "
                f"({q_major_side}, {q_side_ratio:.2f}) | "
                f"d L/R={d_left:2d}/{d_right:2d} "
                f"({d_major_side}, {d_side_ratio:.2f}) | "
                f"opposite={majority_opposite}, strong={strong_opposite}"
            )

    print("\n[summary]")
    print(f"  strong opposite dims = {len(sep_dims)} / {q_pts.shape[1]}")
    print(f"  strong opposite dim ids = {sep_dims}")
    print(f"  query all-left dims  = {len(q_all_left_dims)}")
    print(f"  query all-right dims = {len(q_all_right_dims)}")
    print(f"  doc all-left dims    = {len(d_all_left_dims)}")
    print(f"  doc all-right dims   = {len(d_all_right_dims)}")


def main():
    vocab = np.load(f"{DATA_DIR}/vocab.npy").astype(np.float32)
    queries = np.load(f"{DATA_DIR}/queries.npy", allow_pickle=True)
    dataset = np.load(f"{DATA_DIR}/dataset.npy", allow_pickle=True)

    q_ids = to_ids(queries[QID])
    d_ids = to_ids(dataset[DOC_ID])

    q_pts = vocab[q_ids]
    d_pts = vocab[d_ids]

    cmin = float(vocab.min())
    cmax = float(vocab.max())
    delta = cmax - cmin
    cmin_expanded = cmin - delta

    n_dim = vocab.shape[1]

    print("[data]")
    print(f"  qid={QID}, doc_id={DOC_ID}")
    print(f"  |query|={len(q_ids)}, |doc|={len(d_ids)}")
    print(f"  vocab.shape={vocab.shape}")
    print(f"  cmin={cmin:.6f}")
    print(f"  cmax={cmax:.6f}")
    print(f"  delta={delta:.6f}")
    print()

    s = make_python_random_shifts(delta, n_dim, SEED)

    # base
    mids_base = (cmin_expanded + s + cmax + s) / 2.0
    summarize_side("base random shift", q_pts, d_pts, mids_base)

    # shifted
    for k in KS:
        diff = delta / k
        s_second = np.where(s > delta / 2.0, s - diff, s + diff)
        mids_shifted = (cmin_expanded + s_second + cmax + s_second) / 2.0

        summarize_side(f"shifted delta/{k}", q_pts, d_pts, mids_shifted)


if __name__ == "__main__":
    main()