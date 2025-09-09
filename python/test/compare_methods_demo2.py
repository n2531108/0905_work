import numpy as np
import ot
import time
import ot_estimators as ote


np.random.seed(42)


# ====== ユーティリティ関数 ======
def dist(X, Y):
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True)
    return np.sqrt(np.maximum(XX - 2 * X @ Y.T + YY.T, 0))


def exact_rank_external(query, input_ids, dataset, vocab, topk):
    distances = []
    q_idx, q_weights = zip(*query)
    q_vecs = vocab[list(q_idx)]
    q_weights = np.array(q_weights, dtype=np.float64)
    for idx in input_ids:
        cand = dataset[idx]
        c_idx, c_weights = zip(*cand)
        c_vecs = vocab[list(c_idx)]
        c_weights = np.array(c_weights, dtype=np.float64)
        M = dist(q_vecs, c_vecs).astype(np.float64)  # ← ここを変更
        emd_score = ot.emd2(q_weights, c_weights, M)
        distances.append((emd_score, idx))
    distances.sort()
    top_ids = np.array([x[1] for x in distances[:topk]], dtype=np.int32)
    top_scores = np.array([x[0] for x in distances[:topk]], dtype=np.float32)
    return top_ids, top_scores

def sort_topk(ids_arr, scores_arr):
    order = np.argsort(scores_arr)
    return ids_arr[order], scores_arr[order]

# ====== パラメータ設定 ======
vocab_size = 100
embedding_dim = 10
num_docs = 5000
words_per_doc = 5
topk = 10


# ====== データ生成 ======
vocab = np.random.randn(vocab_size, embedding_dim).astype(np.float32)


def generate_doc():
    ids = np.random.choice(vocab_size, words_per_doc, replace=False)
    weights = np.random.dirichlet(np.ones(words_per_doc))
    return list(zip(ids, weights.astype(np.float32)))


dataset = [generate_doc() for _ in range(num_docs)]
query = generate_doc()
input_ids = list(range(num_docs))
input_ids_np = np.array(input_ids, dtype=np.int32)


# ====== 推定器初期化 ======
estimator = ote.OTEstimators()
estimator.load_vocabulary(vocab)
estimator.load_dataset(dataset)


# ====== Flowtree 実行 ======
output_ids_flow = np.zeros(topk, dtype=np.int32)
output_scores_flow = np.zeros(topk, dtype=np.float32)
t0 = time.time()
estimator.flowtree_rank(query, input_ids_np, output_ids_flow, output_scores_flow, True)
t1 = time.time()
time_flow = t1 - t0
output_ids_flow, output_scores_flow = sort_topk(output_ids_flow, output_scores_flow)

# ====== Quadtree 実行 ======
output_ids_quad = np.zeros(topk, dtype=np.int32)
output_scores_quad = np.zeros(topk, dtype=np.float32)
t0 = time.time()
estimator.quadtree_rank(query, input_ids_np, output_ids_quad, output_scores_quad, True)
t1 = time.time()
time_quad = t1 - t0
output_ids_quad, output_scores_quad = sort_topk(output_ids_quad, output_scores_quad)

# ====== 正確なEMD実行 ======
t0 = time.time()
exact_ids, exact_scores = exact_rank_external(query, input_ids, dataset, vocab, topk)
t1 = time.time()
time_exact = t1 - t0


# ====== 結果出力 ======
print("=== 正確なEMD ===")
print("順序:", exact_ids)
print("スコア:", exact_scores)
print("実行時間: {:.4f} 秒".format(time_exact))


print("\n=== Flowtree ===")
print("順序:", output_ids_flow)
print("スコア:", output_scores_flow)
print("実行時間: {:.4f} 秒".format(time_flow))


print("\n=== Quadtree ===")
print("順序:", output_ids_quad)
print("スコア:", output_scores_quad)
print("実行時間: {:.4f} 秒".format(time_quad))

# 3) 簡易精度指標（Overlap@k）
def overlap_at_k(gt_ids, pred_ids):
    gt, pr = set(map(int, gt_ids)), set(map(int, pred_ids))
    return len(gt & pr) / len(gt)

print("\n=== 精度指標 ===")
print(f"Flowtree Overlap@{topk}: {overlap_at_k(exact_ids, output_ids_flow):.3f}")
print(f"Quadtree Overlap@{topk}: {overlap_at_k(exact_ids, output_ids_quad):.3f}")