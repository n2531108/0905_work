import sys
from collections import Counter
import numpy as np

sys.path.append("/mnt/c/Users/成見/0905_work/native/build")
import ot_estimators_twotree as ote2


def build_children(parents):
    n = len(parents)
    children = [[] for _ in range(n)]
    roots = []
    for node, p in enumerate(parents):
        if p == -1:
            roots.append(node)
        else:
            children[p].append(node)
    return children, roots


def compute_depths(children, roots):
    n = len(children)
    depth = [-1] * n
    stack = []
    for r in roots:
        depth[r] = 0
        stack.append(r)
    while stack:
        u = stack.pop()
        for v in children[u]:
            depth[v] = depth[u] + 1
            stack.append(v)
    return depth


def analyze_tree(parents, leaf):
    children, roots = build_children(parents)
    depth = compute_depths(children, roots)

    child_counts = [len(ch) for ch in children]
    num_nodes = len(parents)
    num_roots = len(roots)
    num_leaves = sum(1 for c in child_counts if c == 0)
    num_internal = sum(1 for c in child_counts if c > 0)
    max_depth = max(depth) if depth else -1

    depth_hist = Counter(depth)
    child_count_hist = Counter(child_counts)

    leaf_depths = [depth[x] for x in leaf]
    leaf_depth_hist = Counter(leaf_depths)

    return {
        "num_nodes": num_nodes,
        "num_roots": num_roots,
        "num_leaves": num_leaves,
        "num_internal": num_internal,
        "max_depth": max_depth,
        "depth_hist": depth_hist,
        "child_count_hist": child_count_hist,
        "leaf_depth_hist": leaf_depth_hist,
    }


def print_hist(name, hist, limit=15):
    print(name)
    for k, v in sorted(hist.items())[:limit]:
        print(f"  {k}: {v}")


def print_tree_summary(name, info):
    print(f"===== {name} =====")
    print(f"num_nodes    = {info['num_nodes']}")
    print(f"num_roots    = {info['num_roots']}")
    print(f"num_leaves   = {info['num_leaves']}")
    print(f"num_internal = {info['num_internal']}")
    print(f"max_depth    = {info['max_depth']}")

    print_hist("child_count_hist (child数 -> node数)", info["child_count_hist"])
    print_hist("depth_hist (深さ -> node数)", info["depth_hist"])
    print_hist("leaf_depth_hist (葉の深さ -> 語彙数)", info["leaf_depth_hist"])
    print()


def main():
    data_dir = "/mnt/c/Users/成見/0905_work/data/otdata_glove50_full"
    vocab = np.load(f"{data_dir}/vocab.npy").astype(np.float32)

    print("vocab.shape =", vocab.shape, "dtype =", vocab.dtype)

    solver = ote2.OTEstimators()
    solver.load_vocabulary(vocab)

    parents1 = solver.get_parents()
    leaf1 = solver.get_leaf()
    parents2 = solver.get_parents_second()
    leaf2 = solver.get_leaf_second()

    print("===== raw sizes =====")
    print("len(parents1) =", len(parents1))
    print("len(parents2) =", len(parents2))
    print("len(leaf1)    =", len(leaf1))
    print("len(leaf2)    =", len(leaf2))
    print()

    info1 = analyze_tree(parents1, leaf1)
    info2 = analyze_tree(parents2, leaf2)

    print_tree_summary("tree1", info1)
    print_tree_summary("tree2", info2)


if __name__ == "__main__":
    main()