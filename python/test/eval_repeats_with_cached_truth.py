#!/usr/bin/env python3
import argparse, time, os, sys, csv
import numpy as np
import algorithms as test  # 既存ファイルを利用（変更不要）

def compute_emd_top1_truth(Q, ids_all, cache_path=None, force=False):
    """
    同一候補集合 ids_all（サイズN）に対して、各クエリ q=0..Q-1 の EMD Top-1 の dataset id を求める。
    cache_path が指定され、存在し、かつ force=False の場合はそれを読み込む。
    """
    if cache_path and (not force) and os.path.exists(cache_path):
        arr = np.load(cache_path)
        # 形状チェック（簡易）
        if arr.shape[0] >= Q:
            return arr[:Q], 0.0, True  # 読み込み時は時間0として返す
    # 計算
    t0 = time.time()
    truth = np.empty(Q, dtype=np.int32)
    for qi in range(Q):
        truth[qi] = test.exact_emd(test.queries[qi], ids_all)
    emd_time = time.time() - t0
    if cache_path:
        np.save(cache_path, truth)
    return truth, emd_time, False

def reinit_solver_for_randomness():
    """
    FlowTree/Quadtree の乱数（load_vocabulary 内のランダムシフト）を変えるため、
    solver を作り直して vocabulary / dataset を再ロードする。
    ※ algorithms.py は変更せず、モジュール内の 'ote' と各グローバルを再利用。
    """
    # test.ote は algorithms.py で `import ot_estimators as ote` しているオブジェクト
    test.solver = test.ote.OTEstimators()
    test.solver.load_vocabulary(test.vocab)            # float32 済
    test.solver.load_dataset(test.dataset_modified)    # [(id, weight)] リスト

def main():
    ap = argparse.ArgumentParser(description="Cache EMD truth once, then evaluate FlowTree/Quadtree with repeats against the same truth.")
    ap.add_argument("--data_dir", default="../data/otdata/", help="path to vocab/dataset/queries/answers")
    ap.add_argument("--topk", type=int, default=100, help="FlowTree/Quadtree で取り出す上位K（表示もKまで）")
    ap.add_argument("--q_max", type=int, default=100, help="評価に使うクエリ数（先頭から）")
    ap.add_argument("--n_max", type=int, default=300, help="評価に使うデータ件数（先頭から）")
    ap.add_argument("--repeats", type=int, default=5, help="FlowTree/Quadtree を繰り返す回数（乱数によるばらつき評価）")
    ap.add_argument("--with_quadtree", action="store_true", help="Quadtree でも指標を計測")
    ap.add_argument("--dump_first", type=int, default=3, help="詳細出力するクエリ数（-1で全件、各repeat=0のみ表示）")
    ap.add_argument("--truth_path", default="", help="EMD Top-1 正解の保存/読み込みパス（.npy）。空なら data_dir に自動命名で保存")
    ap.add_argument("--force_truth", action="store_true", help="truth_path が存在しても EMD を計算し直す")
    args = ap.parse_args()

    # データ読み込み（solver を初期化）
    test.load_data(args.data_dir)

    N = min(args.n_max, len(test.dataset))
    Q = min(args.q_max, len(test.queries))
    K = min(args.topk, N)
    ids_all = np.arange(N, dtype=np.int32)

    k10 = min(10, K)
    k100 = min(100, K)

    # truth の保存先を決める
    if args.truth_path:
        truth_path = args.truth_path
    else:
        base = os.path.abspath(args.data_dir)
        truth_path = os.path.join(base, f"emd_top1_Q{Q}_N{N}.npy")

    # --- EMD 正解（Top-1）を一度だけ用意 ---
    truth_ids, emd_time_once, loaded = compute_emd_top1_truth(Q, ids_all, truth_path, force=args.force_truth)
    if loaded:
        print(f"[truth] loaded from: {truth_path}")
    else:
        print(f"[truth] computed and saved to: {truth_path}")
        print(f"[truth] EMD time (one-shot for Q={Q}, N={N}): {emd_time_once:.2f} sec")

    # --- 繰り返し評価 ---
    ft_top1_runs, ft_rec10_runs, ft_rec100_runs = [], [], []
    qt_top1_runs, qt_rec10_runs, qt_rec100_runs = [], [], []
    t_ft_runs, t_qt_runs = [], []

    def should_print(qi): return args.dump_first < 0 or qi < args.dump_first

    for rep in range(args.repeats):
        # 乱数を変えるために solver を再構築
        reinit_solver_for_randomness()

        ft_top1 = ft_r10 = ft_r100 = 0
        qt_top1 = qt_r10 = qt_r100 = 0
        t_ft = t_qt = 0.0

        for qi in range(Q):
            # --- FlowTree ---
            out_ids_ft    = np.zeros(K, dtype=np.int32)
            out_scores_ft = np.zeros(K, dtype=np.float32)
            t0 = time.time()
            test.solver.flowtree_rank(test.queries_modified[qi], ids_all, out_ids_ft, out_scores_ft, True)
            t_ft += time.time() - t0

            # --- Quadtree（任意） ---
            if args.with_quadtree:
                out_ids_qt    = np.zeros(K, dtype=np.int32)
                out_scores_qt = np.zeros(K, dtype=np.float32)
                t0 = time.time()
                test.solver.quadtree_rank(test.queries_modified[qi], ids_all, out_ids_qt, out_scores_qt, True)
                t_qt += time.time() - t0

            emd_best = truth_ids[qi]

            # --- 集計（FT） ---
            if out_ids_ft[0] == emd_best: ft_top1 += 1
            if emd_best in out_ids_ft[:k10]: ft_r10 += 1
            if emd_best in out_ids_ft[:k100]: ft_r100 += 1

            # --- 集計（QT） ---
            if args.with_quadtree:
                if out_ids_qt[0] == emd_best: qt_top1 += 1
                if emd_best in out_ids_qt[:k10]: qt_r10 += 1
                if emd_best in out_ids_qt[:k100]: qt_r100 += 1

            # --- 表示（repeat=0 のときだけ K 件全部） ---
            if rep == 0 and should_print(qi):
                print(f"\n[rep={rep} q={qi}] EMD_top1 = {int(emd_best)}")
                show = K
                print(f"  [FlowTree top-{show}]")
                for r,(di,sc) in enumerate(zip(out_ids_ft[:show], out_scores_ft[:show]), start=1):
                    mark = "  <-- EMD_TOP1" if di == emd_best else ""
                    print(f"   {r:2d}. id={int(di):6d}  score={float(sc):.6f}{mark}")
                if args.with_quadtree:
                    print(f"  [Quadtree top-{show}]")
                    for r,(di,sc) in enumerate(zip(out_ids_qt[:show], out_scores_qt[:show]), start=1):
                        mark = "  <-- EMD_TOP1" if di == emd_best else ""
                        print(f"   {r:2d}. id={int(di):6d}  score={float(sc):.6f}{mark}")

        # 各リピートの結果を保存
        ft_top1_runs.append(ft_top1/Q)
        ft_rec10_runs.append(ft_r10/Q)
        ft_rec100_runs.append(ft_r100/Q)
        t_ft_runs.append(t_ft/Q)
        if args.with_quadtree:
            qt_top1_runs.append(qt_top1/Q)
            qt_rec10_runs.append(qt_r10/Q)
            qt_rec100_runs.append(qt_r100/Q)
            t_qt_runs.append(t_qt/Q)

        print(f"\n--- repeat {rep+1}/{args.repeats} summary ---")
        print(f"FlowTree: Top-1={ft_top1_runs[-1]:.3f}, R@10={ft_rec10_runs[-1]:.3f}, R@100={ft_rec100_runs[-1]:.3f}, time={t_ft_runs[-1]:.4f}s")
        if args.with_quadtree:
            print(f"Quadtree: Top-1={qt_top1_runs[-1]:.3f}, R@10={qt_rec10_runs[-1]:.3f}, R@100={qt_rec100_runs[-1]:.3f}, time={t_qt_runs[-1]:.4f}s")

    # まとめ（平均±標準偏差）
    def mean_std(x): 
        x = np.array(x, dtype=np.float64)
        return float(x.mean()), float(x.std(ddof=0)) if x.size else (0.0, 0.0)

    m_ft1,s_ft1 = mean_std(ft_top1_runs)
    m_ft10,s_ft10 = mean_std(ft_rec10_runs)
    m_ft100,s_ft100 = mean_std(ft_rec100_runs)
    m_tft,s_tft = mean_std(t_ft_runs)

    print("\n=== FINAL (averaged over repeats) ===")
    print(f"Setup: Q={Q}, N={N}, K={K}, repeats={args.repeats}")
    print(f"EMD truth: path={truth_path}  ({'loaded' if loaded else 'computed'})")
    print("FlowTree:")
    print(f"  Top-1 acc   : {m_ft1:.3f} ± {s_ft1:.3f}")
    print(f"  Recall@10   : {m_ft10:.3f} ± {s_ft10:.3f} {'(K<10 → Recall@%d)'%k10 if k10<10 else ''}")
    print(f"  Recall@100  : {m_ft100:.3f} ± {s_ft100:.3f} {'(K<100 → Recall@%d)'%k100 if k100<100 else ''}")
    print(f"  Time/query  : {m_tft:.4f} ± {s_tft:.4f} sec (FlowTree only)")

    if args.with_quadtree and len(qt_top1_runs)>0:
        m_qt1,s_qt1 = mean_std(qt_top1_runs)
        m_qt10,s_qt10 = mean_std(qt_rec10_runs)
        m_qt100,s_qt100 = mean_std(qt_rec100_runs)
        m_tqt,s_tqt = mean_std(t_qt_runs)
        print("Quadtree:")
        print(f"  Top-1 acc   : {m_qt1:.3f} ± {s_qt1:.3f}")
        print(f"  Recall@10   : {m_qt10:.3f} ± {s_qt10:.3f} {'(K<10 → Recall@%d)'%k10 if k10<10 else ''}")
        print(f"  Recall@100  : {m_qt100:.3f} ± {s_qt100:.3f} {'(K<100 → Recall@%d)'%k100 if k100<100 else ''}")
        print(f"  Time/query  : {m_tqt:.4f} ± {s_tqt:.4f} sec (Quadtree only)")

if __name__ == "__main__":
    main()
