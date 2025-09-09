#!/usr/bin/env python3
import argparse, time, csv, sys
import numpy as np
import algorithms as test  # 既存ファイルを利用

def main():
    ap = argparse.ArgumentParser(description="Evaluate FlowTree against exact EMD, with top-K dump")
    ap.add_argument("--data_dir", default="../data/otdata/", help="path to vocab/dataset/queries/answers")
    ap.add_argument("--topk", type=int, default=10, help="FlowTreeで取り出す上位K")
    ap.add_argument("--q_max", type=int, default=50, help="評価に使うクエリ数（先頭から）")
    ap.add_argument("--n_max", type=int, default=300, help="評価に使うデータ件数（先頭から）")
    ap.add_argument("--dump_first", type=int, default=5, help="画面に詳細を出力するクエリ数（-1で全件）")
    ap.add_argument("--out_csv", default="", help="CSVに全クエリの上位Kを保存するパス（空なら保存しない）")
    args = ap.parse_args()

    # データ読み込み
    test.load_data(args.data_dir)

    N = min(args.n_max, len(test.dataset))
    Q = min(args.q_max, len(test.queries))
    K = min(args.topk, N)
    ids_all = np.arange(N, dtype=np.int32)

    top1_hits = 0
    recall_hits = 0
    t0 = time.time()

    csv_writer = None
    f_csv = None
    if args.out_csv:
        f_csv = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["query_index","rank","dataset_id","flowtree_score","is_emd_top1","emd_top1_id"])

    def should_print(qi):
        return args.dump_first < 0 or qi < args.dump_first

    for qi in range(Q):
        # FlowTree: 上位K
        out_ids    = np.zeros(K, dtype=np.int32)
        out_scores = np.zeros(K, dtype=np.float32)
        test.solver.flowtree_rank(test.queries_modified[qi], ids_all, out_ids, out_scores, True)

        # 厳密EMD: top1（同じ候補集合 ids_all の中で）
        emd_best = test.exact_emd(test.queries[qi], ids_all)

        # 集計
        if out_ids[0] == emd_best:
            top1_hits += 1
        if emd_best in out_ids[:K]:
            recall_hits += 1

        # 画面出力（必要な分だけ）
        if should_print(qi):
            print(f"\n[q={qi}] EMD_top1 = {int(emd_best)}")
            for r,(di,sc) in enumerate(zip(out_ids, out_scores), start=1):
                mark = "  <-- EMD_TOP1" if di == emd_best else ""
                print(f"  {r:2d}. id={int(di):6d}  score={float(sc):.6f}{mark}")

        # CSV には全クエリを書き出す（指定時）
        if csv_writer:
            for r,(di,sc) in enumerate(zip(out_ids, out_scores), start=1):
                csv_writer.writerow([qi, r, int(di), float(sc), int(di==emd_best), int(emd_best)])

    dt = (time.time() - t0) / max(1, Q)
    print("\n=== EVAL (FlowTree vs Exact EMD) ===")
    print(f"Used Q={Q}, N={N}, TOPK={K}")
    print(f"Top-1 accuracy (FT==EMD): {top1_hits/Q:.3f}")
    print(f"Recall@{K} (EMD_top1 in FT_topK): {recall_hits/Q:.3f}")
    print(f"Avg time per query (incl. EMD): {dt:.4f} sec")

    if f_csv:
        f_csv.close()
        print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    main()
