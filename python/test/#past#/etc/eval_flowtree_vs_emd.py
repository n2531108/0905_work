#!/usr/bin/env python3
import argparse, time, csv, sys
import numpy as np
import algorithms as test  # 既存ファイルを利用

def main():
    ap = argparse.ArgumentParser(description="Evaluate FlowTree (and optional Quadtree) vs exact EMD, with multi-K metrics")
    ap.add_argument("--data_dir", default="../data/otdata/", help="path to vocab/dataset/queries/answers")
    ap.add_argument("--topk", type=int, default=100, help="FlowTree/Quadtree で取り出す上位K（表示もKまで）")
    ap.add_argument("--q_max", type=int, default=100, help="評価に使うクエリ数（先頭から）")
    ap.add_argument("--n_max", type=int, default=300, help="評価に使うデータ件数（先頭から）")
    ap.add_argument("--dump_first", type=int, default=5, help="画面に詳細を出力するクエリ数（-1で全件）")
    ap.add_argument("--out_csv", default="", help="CSVに全クエリのTop-K(FlowTreeのみ)を保存するパス（空なら保存しない）")
    ap.add_argument("--with_quadtree", action="store_true", help="Quadtree でも指標を計測")
    args = ap.parse_args()

    test.load_data(args.data_dir)

    N = min(args.n_max, len(test.dataset))
    Q = min(args.q_max, len(test.queries))
    K = min(args.topk, N)
    ids_all = np.arange(N, dtype=np.int32)

    k10 = min(10, K)
    k100 = min(100, K)

    ft_top1_hits = ft_rec10_hits = ft_rec100_hits = 0
    qt_top1_hits = qt_rec10_hits = qt_rec100_hits = 0

    t_ft = t_qt = t_emd = 0.0
    t0_all = time.time()

    csv_writer = None
    f_csv = None
    if args.out_csv:
        f_csv = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["query_index","rank","dataset_id","flowtree_score","is_emd_top1","emd_top1_id"])

    def should_print(qi): return args.dump_first < 0 or qi < args.dump_first

    for qi in range(Q):
        # FlowTree
        out_ids_ft    = np.zeros(K, dtype=np.int32)
        out_scores_ft = np.zeros(K, dtype=np.float32)
        t0 = time.time()
        test.solver.flowtree_rank(test.queries_modified[qi], ids_all, out_ids_ft, out_scores_ft, True)
        t_ft += time.time() - t0

        # Quadtree（任意）
        if args.with_quadtree:
            out_ids_qt    = np.zeros(K, dtype=np.int32)
            out_scores_qt = np.zeros(K, dtype=np.float32)
            t0 = time.time()
            test.solver.quadtree_rank(test.queries_modified[qi], ids_all, out_ids_qt, out_scores_qt, True)
            t_qt += time.time() - t0

        # 厳密EMD（Top-1）
        t0 = time.time()
        emd_best = test.exact_emd(test.queries[qi], ids_all)
        t_emd += time.time() - t0

        # 集計（FlowTree）
        if out_ids_ft[0] == emd_best: ft_top1_hits += 1
        if emd_best in out_ids_ft[:k10]: ft_rec10_hits += 1
        if emd_best in out_ids_ft[:k100]: ft_rec100_hits += 1

        # 集計（Quadtree）
        if args.with_quadtree:
            if out_ids_qt[0] == emd_best: qt_top1_hits += 1
            if emd_best in out_ids_qt[:k10]: qt_rec10_hits += 1
            if emd_best in out_ids_qt[:k100]: qt_rec100_hits += 1

        # 表示（←ここをK件に変更）
        if should_print(qi):
            print(f"\n[q={qi}] EMD_top1 = {int(emd_best)}")
            show = K  # 以前は min(10, K)。指定topkの件数だけ表示するように変更
            print(f"  [FlowTree top-{show}]")
            for r,(di,sc) in enumerate(zip(out_ids_ft[:show], out_scores_ft[:show]), start=1):
                mark = "  <-- EMD_TOP1" if di == emd_best else ""
                print(f"   {r:2d}. id={int(di):6d}  score={float(sc):.6f}{mark}")
            if args.with_quadtree:
                print(f"  [Quadtree top-{show}]")
                for r,(di,sc) in enumerate(zip(out_ids_qt[:show], out_scores_qt[:show]), start=1):
                    mark = "  <-- EMD_TOP1" if di == emd_best else ""
                    print(f"   {r:2d}. id={int(di):6d}  score={float(sc):.6f}{mark}")

        if csv_writer:
            for r,(di,sc) in enumerate(zip(out_ids_ft, out_scores_ft), start=1):
                csv_writer.writerow([qi, r, int(di), float(sc), int(di==emd_best), int(emd_best)])

    # 要約
    elapsed_all = (time.time() - t0_all) / max(1, Q)
    print("\n=== EVAL (FlowTree vs Exact EMD) ===")
    print(f"Used Q={Q}, N={N}, TOPK={K}  (reported: Top-1 / Top-10 / Top-100)")
    print("FlowTree:")
    print(f"  Top-1 acc  : {ft_top1_hits/Q:.3f}")
    print(f"  Recall@10  : {ft_rec10_hits/Q:.3f} {'(K<10 → Recall@%d)'%k10 if k10<10 else ''}")
    print(f"  Recall@100 : {ft_rec100_hits/Q:.3f} {'(K<100 → Recall@%d)'%k100 if k100<100 else ''}")
    if args.with_quadtree:
        print("Quadtree:")
        print(f"  Top-1 acc  : {qt_top1_hits/Q:.3f}")
        print(f"  Recall@10  : {qt_rec10_hits/Q:.3f} {'(K<10 → Recall@%d)'%k10 if k10<10 else ''}")
        print(f"  Recall@100 : {qt_rec100_hits/Q:.3f} {'(K<100 → Recall@%d)'%k100 if k100<100 else ''}")
    print("\nTiming (avg per query):")
    print(f"  FlowTree : {t_ft/max(1,Q):.4f} sec")
    if args.with_quadtree:
        print(f"  Quadtree : {t_qt/max(1,Q):.4f} sec")
    print(f"  EMD      : {t_emd/max(1,Q):.4f} sec")
    print(f"  Total    : {elapsed_all:.4f} sec")

    if f_csv:
        f_csv.close()
        print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    main()


