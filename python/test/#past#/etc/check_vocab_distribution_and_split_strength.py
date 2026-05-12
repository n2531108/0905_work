import numpy as np

DATA_DIR = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"
SEED = 1234
K = 6  # 1/6ずらし。1/24を見たいなら 24 に変更


def main():
    vocab = np.load(f"{DATA_DIR}/vocab.npy").astype(np.float32)
    n, d = vocab.shape

    cmin = float(vocab.min())
    cmax = float(vocab.max())
    delta = cmax - cmin
    cmin_expanded = cmin - delta

    print("[global]")
    print(f"  vocab.shape = {vocab.shape}")
    print(f"  cmin_before_expand = {cmin:.6f}")
    print(f"  cmax               = {cmax:.6f}")
    print(f"  delta              = {delta:.6f}")
    print(f"  delta/{K}           = {delta / K:.6f}")
    print(f"  center_before_shift = {(cmin_expanded + cmax) / 2.0:.6f}")
    print()

    # C++とは完全一致しないが傾向確認用
    rng = np.random.default_rng(SEED)
    s = rng.uniform(0.0, delta, size=d).astype(np.float32)

    diff = delta / K
    s_second = np.where(s > delta / 2.0, s - diff, s + diff)

    mid_base = (cmin_expanded + s + cmax + s) / 2.0
    mid_shift = (cmin_expanded + s_second + cmax + s_second) / 2.0

    rows = []

    for j in range(d):
        x = vocab[:, j]

        q01, q05, q25, q50, q75, q95, q99 = np.quantile(
            x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        std = float(x.std())
        mn = float(x.min())
        mx = float(x.max())

        ratio_base = float(np.mean(x > mid_base[j]))
        ratio_shift = float(np.mean(x > mid_shift[j]))

        # 0.5に近いほど強い分割、0 or 1に近いほど弱い
        strength_base = min(ratio_base, 1.0 - ratio_base)
        strength_shift = min(ratio_shift, 1.0 - ratio_shift)

        rows.append({
            "dim": j,
            "min": mn,
            "q01": q01,
            "q05": q05,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "q95": q95,
            "q99": q99,
            "max": mx,
            "std": std,
            "s": float(s[j]),
            "s_second": float(s_second[j]),
            "mid_base": float(mid_base[j]),
            "mid_shift": float(mid_shift[j]),
            "ratio_base": ratio_base,
            "ratio_shift": ratio_shift,
            "strength_base": strength_base,
            "strength_shift": strength_shift,
        })

    print("[per-dimension distribution and split strength]")
    print(
        "dim | "
        "min q05 q25 med q75 q95 max std | "
        "s mid_base ratio_base strength_base | "
        "s_second mid_shift ratio_shift strength_shift"
    )

    for r in rows:
        print(
            f"{r['dim']:2d} | "
            f"{r['min']: .3f} {r['q05']: .3f} {r['q25']: .3f} "
            f"{r['median']: .3f} {r['q75']: .3f} {r['q95']: .3f} "
            f"{r['max']: .3f} {r['std']: .3f} | "
            f"{r['s']: .3f} {r['mid_base']: .3f} "
            f"{r['ratio_base']: .4f} {r['strength_base']: .4f} | "
            f"{r['s_second']: .3f} {r['mid_shift']: .3f} "
            f"{r['ratio_shift']: .4f} {r['strength_shift']: .4f}"
        )

    print()
    print("[summary: split strength]")
    base_strengths = np.array([r["strength_base"] for r in rows])
    shift_strengths = np.array([r["strength_shift"] for r in rows])

    print(f"  base  mean strength = {base_strengths.mean():.4f}")
    print(f"  shift mean strength = {shift_strengths.mean():.4f}")
    print(f"  base  strong dims strength>=0.10 = {np.sum(base_strengths >= 0.10)} / {d}")
    print(f"  shift strong dims strength>=0.10 = {np.sum(shift_strengths >= 0.10)} / {d}")
    print(f"  base  very strong dims strength>=0.25 = {np.sum(base_strengths >= 0.25)} / {d}")
    print(f"  shift very strong dims strength>=0.25 = {np.sum(shift_strengths >= 0.25)} / {d}")

    print()
    print("[dims where shift makes split much stronger]")
    improved = sorted(rows, key=lambda r: r["strength_shift"] - r["strength_base"], reverse=True)
    for r in improved[:15]:
        print(
            f"dim={r['dim']:2d} "
            f"s={r['s']:.3f} -> s_second={r['s_second']:.3f} | "
            f"mid={r['mid_base']:.3f} -> {r['mid_shift']:.3f} | "
            f"ratio={r['ratio_base']:.4f} -> {r['ratio_shift']:.4f} | "
            f"strength={r['strength_base']:.4f} -> {r['strength_shift']:.4f}"
        )

    print()
    print("[s range analysis]")
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, delta + 1e-6]
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, r in enumerate(rows) if lo <= r["s"] < hi]
        if not idx:
            continue
        b = base_strengths[idx]
        sh = shift_strengths[idx]
        print(
            f"  s in [{lo:.1f}, {hi:.1f}): "
            f"count={len(idx):2d}, "
            f"base_strength_mean={b.mean():.4f}, "
            f"shift_strength_mean={sh.mean():.4f}"
        )

    print()
    print("[interpretation]")
    print("  ratio = vocab[:, dim] > mid となる点の割合")
    print("  strength = min(ratio, 1-ratio)")
    print("  strength が 0 に近いほど分割は弱い")
    print("  strength が 0.5 に近いほど分割は強い")
    print("  どの s で分割が強まるかは、mid がその次元の分布の中心付近に来るかで決まる")


if __name__ == "__main__":
    main()