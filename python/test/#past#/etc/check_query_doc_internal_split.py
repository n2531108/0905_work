import numpy as np


DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"

QID = 1
DOC_ID = 1
SEED = 1234
KS = [6, 12, 24, 48, 100]


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
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, delta, size=d).astype(np.float32)


def summarize_internal_split(label, q_pts, d_pts, mids):
    q_ratios = []
    d_ratios = []
    mixed_q_dims = []
    mixed_d_dims = []
    mixed_both_dims = []

    print(f"\n[{label}]")

    for j in range(q_pts.shape[1]):
        q_vals = q_pts[:, j]
        d_vals = d_pts[:, j]
        mid = mids[j]

        q_left = np.sum(q_vals <= mid)
        q_right = np.sum(q_vals > mid)
        d_left = np.sum(d_vals <= mid)
        d_right = np.sum(d_vals > mid)

        # 1.0 に近いほど片側に固まる
        # 0.5 に近いほど左右に割れている
        q_major_ratio = max(q_left, q_right) / len(q_vals)
        d_major_ratio = max(d_left, d_right) / len(d_vals)

        q_ratios.append(q_major_ratio)
        d_ratios.append(d_major_ratio)

        q_mixed = q_left > 0 and q_right > 0
        d_mixed = d_left > 0 and d_right > 0

        if q_mixed:
            mixed_q_dims.append(j)
        if d_mixed:
            mixed_d_dims.append(j)
        if q_mixed and d_mixed:
            mixed_both_dims.append(j)

        print(
            f"dim={j:2d} mid={mid: .4f} | "
            f"query L/R={q_left:2d}/{q_right:2d}, "
            f"major={q_major_ratio:.2f} | "
            f"doc L/R={d_left:2d}/{d_right:2d}, "
            f"major={d_major_ratio:.2f}"
        )

    q_ratios = np.array(q_ratios)
    d_ratios = np.array(d_ratios)

    print("\n[summary]")
    print(f"  query mixed dims      = {len(mixed_q_dims)} / {q_pts.shape[1]}")
    print(f"  doc mixed dims        = {len(mixed_d_dims)} / {q_pts.shape[1]}")
    print(f"  both mixed dims       = {len(mixed_both_dims)} / {q_pts.shape[1]}")
    print(f"  query mean major rate = {q_ratios.mean():.3f}")
    print(f"  doc mean major rate   = {d_ratios.mean():.3f}")
    print(f"  query median major    = {np.median(q_ratios):.3f}")
    print(f"  doc median major      = {np.median(d_ratios):.3f}")

    print("\n[interpretation]")
    print("  mixed dims が多いほど、その集合は境界で内部分裂している。")
    print("  major rate が 1.0 に近いほど片側に固まっている。")
    print("  major rate が 0.5 に近いほど左右に割れている。")


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
    d = vocab.shape[1]

    print("[data]")
    print(f"  qid={QID}, doc_id={DOC_ID}")
    print(f"  |query|={len(q_ids)}, |doc|={len(d_ids)}")
    print(f"  delta={delta:.6f}")

    s = make_python_random_shifts(delta, d, SEED)

    mids_base = (cmin_expanded + s + cmax + s) / 2.0
    summarize_internal_split("base random shift", q_pts, d_pts, mids_base)

    for k in KS:
        diff = delta / k
        s_second = np.where(s > delta / 2.0, s - diff, s + diff)
        mids_shifted = (cmin_expanded + s_second + cmax + s_second) / 2.0
        summarize_internal_split(f"shifted delta/{k}", q_pts, d_pts, mids_shifted)


if __name__ == "__main__":
    main()