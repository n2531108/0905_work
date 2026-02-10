import os
import numpy as np

data_dir="/mnt/c/Users/成見/0905_work/tmp_flowtree_demo"
V = np.load(os.path.join(data_dir,"vocab.npy"))

S = [47,83,49,38,18,10,0,86,67,42]

fA = {47:0, 83:2, 18:3, 49:5, 42:6, 38:8, 10:12, 0:14, 86:15, 67:19}
fB = {47:0, 83:5, 10:6, 49:8, 38:9, 18:14, 0:15, 42:16, 86:19, 67:20}

def l2(u,v):
    return float(np.linalg.norm(V[u]-V[v]))

def nn_by(f, u):
    # (1D距離, L2, v) の辞書順最小を最近傍とする（同距離ならL2小さい方）
    cand=[]
    for v in S:
        if v==u: continue
        cand.append((abs(f[u]-f[v]), l2(u,v), v))
    cand.sort()
    d, dist, v = cand[0]
    return v, d, dist

def nn_by_l2(u):
    # (L2, v) の辞書順最小を L2 最近傍とする
    cand=[]
    for v in S:
        if v==u: continue
        cand.append((l2(u,v), v))
    cand.sort()
    dist, v = cand[0]
    return v, dist

both_cnt = 0
diff_cnt = 0

# 追加サマリー用
a_eq_l2_cnt = 0
b_eq_l2_cnt = 0
all_three_cnt = 0

print("=== Per-vertex neighbor list (sorted by dA; marks show A/B/L2 nearest) ===")
print("Mark legend: A* (A-NN), B* (B-NN), L2* (L2-NN), combos like ABL2*")

for u in S:
    a_v, a_d, a_l2 = nn_by(fA, u)
    b_v, b_d, b_l2 = nn_by(fB, u)
    l2_v, l2_d = nn_by_l2(u)

    status = "BOTH" if a_v == b_v else "DIFF"
    if status == "BOTH": both_cnt += 1
    else: diff_cnt += 1

    # 追加サマリー用カウント
    if a_v == l2_v: a_eq_l2_cnt += 1
    if b_v == l2_v: b_eq_l2_cnt += 1
    if (a_v == b_v) and (a_v == l2_v): all_three_cnt += 1

    print(f"\n[u={u:>2d}] status={status}  "
          f"A-NN={a_v:>2d}(dA={a_d},L2={a_l2:.3f})  "
          f"B-NN={b_v:>2d}(dB={b_d},L2={b_l2:.3f})  "
          f"L2-NN={l2_v:>2d}(L2={l2_d:.3f})")

    lst=[]
    for v in S:
        if v==u: continue
        da = abs(fA[u]-fA[v])
        db = abs(fB[u]-fB[v])
        dist = l2(u,v)
        lst.append((da, dist, db, v))
    lst.sort(key=lambda t: (t[0], t[1], t[2]))  # dA→L2→dB

    print(" rank   mark    v   dA  dB   L2")
    for i,(da,dist,db,v) in enumerate(lst, start=1):
        tags = []
        if v == a_v:  tags.append("A")
        if v == b_v:  tags.append("B")
        if v == l2_v: tags.append("L2")
        mark = ("".join(tags) + "*") if tags else ""
        print(f" {i:>3d} {mark:>6s} {v:>3d} {da:>3d} {db:>3d} {dist:>6.3f}")

# 全体集計
n = len(S)
both_ratio = both_cnt / n * 100.0
diff_ratio = diff_cnt / n * 100.0
a_eq_l2_ratio = a_eq_l2_cnt / n * 100.0
b_eq_l2_ratio = b_eq_l2_cnt / n * 100.0
all_three_ratio = all_three_cnt / n * 100.0

print("\n=== Summary over all vertices ===")
print(f"#vertices = {n}")
print(f"BOTH: {both_cnt} ({both_ratio:.1f}%)  |  DIFF: {diff_cnt} ({diff_ratio:.1f}%)")
print(f"A-NN == L2-NN: {a_eq_l2_cnt} ({a_eq_l2_ratio:.1f}%)")
print(f"B-NN == L2-NN: {b_eq_l2_cnt} ({b_eq_l2_ratio:.1f}%)")
print(f"A-NN == B-NN == L2-NN: {all_three_cnt} ({all_three_ratio:.1f}%)")
