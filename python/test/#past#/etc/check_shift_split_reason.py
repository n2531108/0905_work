import numpy as np
from collections import Counter


DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"
SEED = 1234
DIMS_TO_SHOW = 10
KS = [6, 12, 24, 48, 100]


def make_cpp_like_shifts(delta, d, seed=1234):
    """
    注意:
    C++ の mt19937_64 + uniform_real_distribution<float> と
    Python/numpy の乱数は完全一致しません。
    ここでは傾向確認用です。
    厳密一致させたい場合は C++ 側で s を出力してください。
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, delta, size=d).astype(np.float32)


def root_codes_and_cell_sizes(vocab, cmin_expanded, cmax, shifts):
    d = vocab.shape[1]
    mids = ((cmin_expanded + shifts) + (cmax + shifts)) / 2.0

    # 各点について、50次元の > mid 判定を bit code にする
    bits = vocab > mids[None, :]

    # Python上で扱いやすいように bytes 化
    packed = np.packbits(bits.astype(np.uint8), axis=1)
    codes = [row.tobytes() for row in packed]

    counter = Counter(codes)
    sizes = np.array(list(counter.values()), dtype=np.int32)
    sizes.sort()

    return mids, sizes


def summarize_sizes(label, sizes):
    n = len(sizes)
    print(f"[root cell size] {label}")
    print(f"  parts      = {n}")
    print(f"  min        = {sizes[0]}")
    print(f"  median     = {int(np.median(sizes))}")
    print(f"  mean       = {sizes.mean():.3f}")
    print(f"  p90        = {int(np.quantile(sizes, 0.90))}")
    print(f"  p99        = {int(np.quantile(sizes, 0.99))}")
    print(f"  max        = {sizes[-1]}")
    print()


def main():
    vocab = np.load(f"{DATA_DIR}/vocab.npy").astype(np.float32)
    n, d = vocab.shape

    cmin = float(vocab.min())
    cmax = float(vocab.max())
    delta = cmax - cmin
    cmin_expanded = cmin - delta

    print("[data]")
    print(f"  vocab.shape = {vocab.shape}")
    print(f"  cmin        = {cmin:.6f}")
    print(f"  cmax        = {cmax:.6f}")
    print(f"  delta       = {delta:.6f}")
    print()

    # 各次元の分布
    std = vocab.std(axis=0)
    dim_min = vocab.min(axis=0)
    dim_max = vocab.max(axis=0)
    dim_range = dim_max - dim_min
    q25 = np.quantile(vocab, 0.25, axis=0)
    q50 = np.quantile(vocab, 0.50, axis=0)
    q75 = np.quantile(vocab, 0.75, axis=0)

    print("[dimension scale]")
    print(f"  mean(std)        = {std.mean():.6f}")
    print(f"  median(std)      = {np.median(std):.6f}")
    print(f"  mean(dim_range)  = {dim_range.mean():.6f}")
    print(f"  median(dim_range)= {np.median(dim_range):.6f}")
    print()

    print("[shift size compared with dimension scale]")
    for k in KS:
        shift = delta / k
        print(f"  delta/{k:<3} = {shift:.6f} | "
              f"/mean(std)={shift/std.mean():.3f}, "
              f"/median(std)={shift/np.median(std):.3f}, "
              f"/mean(range)={shift/dim_range.mean():.3f}, "
              f"/median(range)={shift/np.median(dim_range):.3f}")
    print()

    # 元のランダム shift
    s = make_cpp_like_shifts(delta, d, SEED)

    print("[root split comparison]")
    print("  note: Python random shifts are trend-check only, not bit-identical to C++.")
    print()

    # 元 tree
    mids_base, sizes_base = root_codes_and_cell_sizes(vocab, cmin_expanded, cmax, s)
    summarize_sizes("base random shift", sizes_base)

    for k in KS:
        diff = delta / k

        s_second = s.copy()
        s_second = np.where(s > delta / 2.0, s - diff, s + diff)

        mids_shifted, sizes_shifted = root_codes_and_cell_sizes(
            vocab, cmin_expanded, cmax, s_second
        )

        summarize_sizes(f"shifted by delta/{k}", sizes_shifted)

        print(f"[mid position check] shifted delta/{k}")
        for j in range(min(DIMS_TO_SHOW, d)):
            mid = mids_shifted[j]
            in_iqr = q25[j] <= mid <= q75[j]
            print(
                f"  dim={j:2d} "
                f"mid={mid: .6f} "
                f"q25={q25[j]: .6f} "
                f"median={q50[j]: .6f} "
                f"q75={q75[j]: .6f} "
                f"std={std[j]: .6f} "
                f"in_IQR={in_iqr}"
            )
        print()

    print("[interpretation guide]")
    print("  1. delta/k が std の何倍かを見る。大きいほど境界移動が大きい。")
    print("  2. parts が増えるほど、root でセルが細かく分かれている。")
    print("  3. median cell size が小さいほど、小さいセルが大量発生している。")
    print("  4. mid が q25〜q75 に入る次元が多いほど、密集領域を切っている可能性が高い。")


if __name__ == "__main__":
    main()