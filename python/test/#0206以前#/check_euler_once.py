#!/usr/bin/env python3
import sys, re, random

def grab_line(tag, txt):
    m = re.search(rf'^{re.escape("#"+tag)}\s+(.+)$', txt, flags=re.M)
    if not m: return None
    return list(map(int, m.group(1).split()))

def main():
    if len(sys.argv) >= 2:
        log_path = sys.argv[1]
    else:
        # デフォルト：あなたの環境の既定パス
        log_path = "/mnt/c/Users/成見/0905_work/tmp_flowtree_demo/euler_debug_once.txt"

    with open(log_path, encoding="utf-8") as f:
        txt = f.read()

    # ブロックが複数入っている場合に備え、最後のブロックを使う
    blocks = [m.start() for m in re.finditer(r'^#PACKED_NODES\s+\d+', txt, flags=re.M)]
    if not blocks:
        sys.exit("ERROR: #PACKED_NODES が見つかりません。--dump-euler で出力したログか確認してください。")
    txt = txt[blocks[-1]:]

    Nline = re.search(r'^#PACKED_NODES\s+(\d+)', txt, flags=re.M)
    N = int(Nline.group(1))
    root = grab_line("ROOT", txt)[0]
    tin   = grab_line("TIN", txt)
    tout  = grab_line("TOUT", txt)
    depth = grab_line("DEPTH", txt)
    first = grab_line("FIRST", txt)
    euler = grab_line("EULER", txt)
    unleaf = grab_line("UNLEAF", txt)  # ない場合もある

    # 1) 配列整合性
    assert len(tin)==len(tout)==len(depth)==len(first)==N, "配列長が不一致"
    assert all(0 <= first[u] < len(euler) for u in range(N)), "first_occ 範囲外"
    for u in range(N):
        assert euler[first[u]] == u, f"first_occ 不整合 at u={u}"

    # 2) 区間（部分木＝連続区間）
    for u in range(N):
        assert 0 <= tin[u] < tout[u], f"tin/tout 不正 at u={u}"

    def is_ancestor(u,v):
        return tin[u] <= tin[v] and tout[v] <= tout[u]

    # ランダムに矛盾がないか確認
    for _ in range(min(200, max(50, N))):
        u, v = random.randrange(N), random.randrange(N)
        if is_ancestor(u,v) and is_ancestor(v,u):
            assert u==v, f"自己以外で相互祖先: u={u}, v={v}"

    # 3) 根の性質
    assert tin[root] == min(tin), "根の tin が最小ではない可能性"
    assert tout[root] == max(tout), "根の tout が最大ではない可能性"

    # 4) 目安：Euler 長
    L = len(euler)
    print(f"[OK] N={N}, euler_len={L} (参考: 2*N-1={2*N-1})")

    # 5) 葉の性質
    subsize = [tout[u]-tin[u] for u in range(N)]
    leafs = [u for u in range(N) if subsize[u]==1]
    print("subtree_size(先頭10):", subsize[:10])
    print("葉(候補) 数:", len(leafs))

    # 6) 葉→語彙ID（UNLEAFがあれば）
    if unleaf and len(unleaf)==N:
        leaf_vocab = [unleaf[u] for u in leafs]
        print("葉の語彙ID(先頭20):", leaf_vocab[:20])

    print("All invariants passed: 1D embedding via Euler looks consistent.")

if __name__ == "__main__":
    main()
