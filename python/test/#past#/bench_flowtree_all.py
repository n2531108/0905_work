#!/usr/bin/env python3
import argparse, time, numpy as np
import algorithms as test  # 既存の algorithms.py を利用（変更不要）

def main():
    ap = argparse.ArgumentParser(description="FlowTree over ALL dataset; measure time")
    ap.add_argument("--data_dir", default="../data/otdata/", help="vocab/dataset/queries/answers のフォルダ")
    ap.add_argument("--q_start", type=int, default=0, help="開始クエリindex")
    ap.add_argument("--q_count", type=int, default=1, help="計測するクエリ数（連続）")
    ap.add_argument("--topk", type=int, default=50, help="FlowTreeで返す上位K（計測には影響ほぼ無し）")
    ap.add_argument("--show_top", type=int, default=5, help="各クエリで表示する上位件数（IDとスコア）")
    args = ap.parse_args()

    # データ読み込み
    test.load_data(args.data_dir)
    N = len(test.dataset)
    Q = len(test.queries)
    print(f"dataset N = {N}, queries Q = {Q}")
    if Q == 0 or N == 0:
        raise SystemExit("データが空です")

    q0 = max(0, min(args.q_start, Q-1))
    q1 = min(q0 + max(1, args.q_count), Q)
    K = min(max(1, args.topk), N)
    show = min(max(0, args.show_top), K)

    ids_all = np.arange(N, dtype=np.int32)

    # 1回ウォームアップ（計測外）
    _ids = np.zeros(K, dtype=np.int32)
    _sc  = np.zeros(K, dtype=np.float32)
    test.solver.flowtree_rank(test.queries_modified[q0], ids_all, _ids, _sc, True)

    # 本計測
    total = 0.0
    for qi in range(q0, q1):
        out_ids = np.zeros(K, dtype=np.int32)
        out_sc  = np.zeros(K, dtype=np.float32)
        t0 = time.perf_counter()
        test.solver.flowtree_rank(test.queries_modified[qi], ids_all, out_ids, out_sc, True)
        dt = time.perf_counter() - t0
        total += dt
        print(f"\n[q={qi}] FlowTree over ALL {N} docs -> top-{K} in {dt:.4f} sec")
        for r,(di,sc) in enumerate(zip(out_ids[:show], out_sc[:show]), 1):
            print(f"  {r:2d}. id={int(di):6d}  ft_score={float(sc):.6f}")

    avg = total / (q1 - q0)
    print(f"\n=== SUMMARY ===")
    print(f"queries measured: {q1 - q0} (from {q0} to {q1-1})")
    print(f"topK: {K}")
    print(f"avg time per query (FlowTree only): {avg:.4f} sec")

if __name__ == "__main__":
    main()
