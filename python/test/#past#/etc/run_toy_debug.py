import sys
import numpy as np

sys.path.append("/mnt/c/Users/成見/0905_work/native/build")
import ot_estimators_twotree as ote2


def ids_to_uniform_measure(ids):
    w = 1.0 / len(ids)
    return [(int(v), float(w)) for v in ids]


def run_case(case_name, vocab_1d, query_ids, doc_ids):
    print("\n" + "=" * 100)
    print(f"[python] {case_name}")
    print("=" * 100)

    vocab = np.array(vocab_1d, dtype=np.float32).reshape(-1, 1)
    query = ids_to_uniform_measure(query_ids)
    doc = ids_to_uniform_measure(doc_ids)

    print("[python] vocab:")
    for i, x in enumerate(vocab_1d):
        print(f"  vid={i}  x={x:.4f}")

    print("[python] query_ids =", query_ids)
    print("[python] doc_ids   =", doc_ids)

    solver = ote2.OTEstimators()
    solver.load_vocabulary(vocab)
    solver.load_dataset([doc])

    input_ids = np.array([0], dtype=np.int32)
    output_ids = np.zeros(1, dtype=np.int32)
    output_scores = np.zeros(1, dtype=np.float32)

    # ここで C++ 側の [debug] 出力がそのまま出る
    solver.flowtree_rank(query, input_ids, output_ids, output_scores, True)

    print("[python] flowtree score =", float(output_scores[0]))
    print("[python] top doc id     =", int(output_ids[0]))


def main():
    vocab_1d = [
        0.05, 0.10, 0.15, 0.20,
        0.45, 0.50, 0.55,
        0.80, 0.85, 0.90,
    ]

    run_case(
        case_name="case0 (left local)",
        vocab_1d=vocab_1d,
        query_ids=[1, 2],
        doc_ids=[0, 3],
    )

    run_case(
        case_name="case1 (left-middle-right)",
        vocab_1d=vocab_1d,
        query_ids=[1, 2, 8],
        doc_ids=[0, 5, 9],
    )


if __name__ == "__main__":
    main()