import sys
import numpy as np

sys.path.append("/mnt/c/Users/成見/0905_work/native/build")
import ot_estimators_twotree as ote2


def ids_to_uniform_measure(ids):
    if len(ids) == 0:
        return []
    w = 1.0 / len(ids)
    return [(int(v), float(w)) for v in ids]


def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_vocab(vocab_1d):
    print_header("VOCAB (original 1D positions)")
    print(f"{'vid':>4}  {'orig_x':>10}")
    for vid, x in enumerate(vocab_1d):
        print(f"{vid:4d}  {x:10.4f}")


def print_measure(name, ids, vocab_1d):
    print_header(name)
    print(f"{'vid':>4}  {'orig_x':>10}")
    for vid in ids:
        print(f"{vid:4d}  {vocab_1d[vid]:10.4f}")


def build_role_table(plus_pts, minus_pts):
    """
    plus_pts/minus_pts:
      list of tuples (vid, packed_x, mass)
    return:
      vid -> (role, packed_x, mass)
    """
    out = {}
    for vid, x, mass in plus_pts:
        out[int(vid)] = ("+", int(x), float(mass))
    for vid, x, mass in minus_pts:
        out[int(vid)] = ("-", int(x), float(mass))
    return out


def print_embedding_table(title, vocab_1d, plus_pts, minus_pts):
    """
    packed_x 順に並べて、元座標と役割をまとめて表示
    """
    role_map = build_role_table(plus_pts, minus_pts)

    rows = []
    for vid, (role, px, mass) in role_map.items():
        rows.append((px, vid, role, vocab_1d[vid], mass))

    rows.sort(key=lambda t: (t[0], t[1]))

    print_header(title)
    print(f"{'idx':>4}  {'packed_x':>8}  {'vid':>4}  {'role':>4}  {'orig_x':>10}  {'mass':>12}")
    for i, (px, vid, role, ox, mass) in enumerate(rows):
        print(f"{i:4d}  {px:8d}  {vid:4d}  {role:>4}  {ox:10.4f}  {mass:12.9f}")


def print_points_table(title, pts, vocab_1d):
    """
    pts: list of tuples (vid, packed_x, mass)
    packed_x 順に表示
    """
    pts_sorted = sorted([(int(v), int(x), float(m)) for v, x, m in pts], key=lambda t: (t[1], t[0]))

    print_header(title)
    print(f"{'idx':>4}  {'vid':>4}  {'orig_x':>10}  {'packed_x':>8}  {'mass':>12}")
    for i, (vid, px, mass) in enumerate(pts_sorted):
        print(f"{i:4d}  {vid:4d}  {vocab_1d[vid]:10.4f}  {px:8d}  {mass:12.9f}")


def print_side_by_side_tree_compare(vocab_1d, plus1, minus1, plus2, minus2):
    """
    同じ vid が tree1 / tree2 でどう埋め込まれたかを見る
    """
    d1 = build_role_table(plus1, minus1)
    d2 = build_role_table(plus2, minus2)

    vids = sorted(set(d1.keys()) | set(d2.keys()))

    print_header("TREE1 / TREE2 embedding compare by vid")
    print(
        f"{'vid':>4}  {'orig_x':>10} | "
        f"{'t1_role':>7} {'t1_x':>6} {'t1_mass':>12} | "
        f"{'t2_role':>7} {'t2_x':>6} {'t2_mass':>12}"
    )

    for vid in vids:
        ox = vocab_1d[vid]

        r1 = d1.get(vid, ("", -1, -1.0))
        r2 = d2.get(vid, ("", -1, -1.0))

        print(
            f"{vid:4d}  {ox:10.4f} | "
            f"{r1[0]:>7} {r1[1]:6d} {r1[2]:12.9f} | "
            f"{r2[0]:>7} {r2[1]:6d} {r2[2]:12.9f}"
        )


def run_case(case_name, vocab_1d, query_ids, doc_ids):
    print_header(f"CASE: {case_name}")

    print_measure("QUERY ids / original positions", query_ids, vocab_1d)
    print_measure("DOC ids / original positions", doc_ids, vocab_1d)

    vocab = np.array(vocab_1d, dtype=np.float32).reshape(-1, 1)
    query = ids_to_uniform_measure(query_ids)
    doc = ids_to_uniform_measure(doc_ids)

    solver = ote2.OTEstimators()
    solver.load_vocabulary(vocab)
    solver.load_dataset([doc])

    input_ids = np.array([0], dtype=np.int32)
    output_ids = np.zeros(1, dtype=np.int32)
    output_scores = np.zeros(1, dtype=np.float32)

    solver.flowtree_rank(query, input_ids, output_ids, output_scores, True)

    # tree1
    packed_x1 = solver.get_last_packed_x()
    plus1 = solver.get_last_plus_pts()
    minus1 = solver.get_last_minus_pts()

    # tree2
    packed_x2 = solver.get_last_packed_x_second()
    plus2 = solver.get_last_plus_pts_second()
    minus2 = solver.get_last_minus_pts_second()

    print_header("BASIC SUMMARY")
    print("[tree1]")
    print("  len(packed_x1) =", len(packed_x1))
    print("  len(plus1)     =", len(plus1))
    print("  len(minus1)    =", len(minus1))
    print("[tree2]")
    print("  len(packed_x2) =", len(packed_x2))
    print("  len(plus2)     =", len(plus2))
    print("  len(minus2)    =", len(minus2))

    print_header("PACKED_X HEAD")
    print("tree1:", packed_x1[:60])
    print("tree2:", packed_x2[:60])

    print_points_table("TREE1 plus_pts sorted by packed_x", plus1, vocab_1d)
    print_points_table("TREE1 minus_pts sorted by packed_x", minus1, vocab_1d)

    print_points_table("TREE2 plus_pts sorted by packed_x", plus2, vocab_1d)
    print_points_table("TREE2 minus_pts sorted by packed_x", minus2, vocab_1d)

    print_embedding_table("TREE1 merged embedding view (packed_x order)", vocab_1d, plus1, minus1)
    print_embedding_table("TREE2 merged embedding view (packed_x order)", vocab_1d, plus2, minus2)

    print_side_by_side_tree_compare(vocab_1d, plus1, minus1, plus2, minus2)


def main():
    # 1次元・10語彙
    vocab_1d = [
        0.05, 0.10, 0.15, 0.20,
        0.45, 0.50, 0.55,
        0.80, 0.85, 0.90,
    ]

    print_vocab(vocab_1d)

    # テストケース0: 左群だけの小ケース
    run_case(
        case_name="case0 (left local)",
        vocab_1d=vocab_1d,
        query_ids=[1, 2],
        doc_ids=[0, 3],
    )

    # テストケース1: 左・中・右をまたぐケース
    run_case(
        case_name="case1 (left-middle-right)",
        vocab_1d=vocab_1d,
        query_ids=[1, 2, 8],
        doc_ids=[0, 5, 9],
    )


if __name__ == "__main__":
    main()