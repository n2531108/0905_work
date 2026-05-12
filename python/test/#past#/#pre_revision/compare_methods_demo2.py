#!/usr/bin/env python3
import os, time
import numpy as np
import ot
import ot_estimators as ote

# ========= ユーティリティ =========
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
        M = dist(q_vecs, c_vecs).astype(np.float64)
        emd_score = ot.emd2(q_weights, c_weights, M)
        distances.append((emd_score, idx))
    distances.sort()
    top_ids = np.array([x[1] for x in distances[:topk]], dtype=np.int32)
    top_scores = np.array([x[0] for x in distances[:topk]], dtype=np.float32)
    return top_ids, top_scores

def sort_topk(ids_arr, scores_arr):
    order = np.argsort(scores_arr)
    return ids_arr[order], scores_arr[order]

def overlap_at_k(gt_ids, pred_ids):
    gt, pr = set(map(int, gt_ids)), set(map(int, pred_ids))
    return len(gt & pr) / len(gt)

# ========= パラメータ =========
vocab_size = 100
embedding_dim = 10
num_docs = 5000
words_per_doc = 5
topk = 10

# ========= データ生成 =========
np.random.seed(42)
vocab = np.random.randn(vocab_size, embedding_dim).astype(np.float32)

def generate_doc():
    ids = np.random.choice(vocab_size, words_per_doc, replace=False)
    weights = np.random.dirichlet(np.ones(words_per_doc))
    return list(zip(ids, weights.astype(np.float32)))

dataset = [generate_doc() for _ in range(num_docs)]
query = generate_doc()
input_ids = np.arange(num_docs, dtype=np.int32)

# ========= 推定器初期化（既存実装） =========
estimator = ote.OTEstimators()
estimator.load_vocabulary(vocab)
estimator.load_dataset(dataset)

# ========= Flowtree =========
output_ids_flow = np.zeros(topk, dtype=np.int32)
output_scores_flow = np.zeros(topk, dtype=np.float32)
t0 = time.time()
estimator.flowtree_rank(query, input_ids, output_ids_flow, output_scores_flow, True)
time_flow = time.time() - t0
output_ids_flow, output_scores_flow = sort_topk(output_ids_flow, output_scores_flow)

# ========= Quadtree =========
output_ids_quad = np.zeros(topk, dtype=np.int32)
output_scores_quad = np.zeros(topk, dtype=np.float32)
t0 = time.time()
estimator.quadtree_rank(query, input_ids, output_ids_quad, output_scores_quad, True)
time_quad = time.time() - t0
output_ids_quad, output_scores_quad = sort_topk(output_ids_quad, output_scores_quad)

# ========= 厳密 EMD =========
t0 = time.time()
exact_ids, exact_scores = exact_rank_external(query, input_ids, dataset, vocab, topk)
time_exact = time.time() - t0

# ========= 結果 =========
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

print("\n=== 精度指標 ===")
print(f"Flowtree Overlap@{topk}: {overlap_at_k(exact_ids, output_ids_flow):.3f}")
print(f"Quadtree Overlap@{topk}: {overlap_at_k(exact_ids, output_ids_quad):.3f}")

# ========= C++ 砂場連携のための保存 =========
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
outdir = os.path.join(project_root, "tmp_flowtree_demo")
os.makedirs(outdir, exist_ok=True)

# vocab
np.save(os.path.join(outdir, "vocab.npy"), vocab.astype(np.float32))

# dataset を (ids, weights, offsets) に平坦化
ids_flat, w_flat, offsets = [], [], [0]
for doc in dataset:
    ids, ws = zip(*doc)
    ids_flat.extend(ids)
    w_flat.extend(ws)
    offsets.append(len(ids_flat))
np.savez(os.path.join(outdir, "dataset.npz"),
         ids=np.array(ids_flat, dtype=np.int32),
         weights=np.array(w_flat, dtype=np.float32),
         offsets=np.array(offsets, dtype=np.int32))

# query
q_ids, q_ws = zip(*query)
np.savez(os.path.join(outdir, "query.npz"),
         ids=np.array(q_ids, dtype=np.int32),
         weights=np.array(q_ws, dtype=np.float32))

# input_ids
np.save(os.path.join(outdir, "input_ids.npy"), input_ids.astype(np.int32))

with open(os.path.join(outdir, "meta.txt"), "w") as f:
    f.write(f"topk={topk}\n")

print(f"\n[dumped] {os.path.relpath(outdir, project_root)} に保存しました")

# ========= 砂場（C++）を subprocess で呼ぶ =========
from sandbox_runner import FlowtreeSandboxRunner

exe_path = os.path.join(project_root, "cpp/test/flowtree_sandbox")
runner = FlowtreeSandboxRunner(exe_path, outdir)
cpp_ids, cpp_scores = runner.topk(topk)

print("\n=== Sandbox-MeanDistance (C++ via subprocess) ===")
print("順序:", cpp_ids)
print("スコア:", cpp_scores)
print(f"Overlap@{topk} vs exact:", f"{overlap_at_k(exact_ids, cpp_ids):.3f}")
