#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed1d_algoA_demo.py

Algorithm A（論文の“1D tree embedding”擬似コード）をそのまま実装し、
小さな手作り木 or FlowTree サンドボックスのダンプ（--from-adj）に対して
葉の一次元座標 f(x) を計算・表示するデモ。

対応モード:
  --mode A | Aprime | C | Cprime | Cdoubleprime | D | Dprime | Dprime_manual
  --from-adj <adj_debug_once.txt>   # flowtree_sandbox --dump-adj の出力を読む
（--from-euler は省略。必要になれば追加）

使い方例:
  python3 embed1d_algoA_demo.py --mode D
  python3 embed1d_algoA_demo.py --mode Dprime_manual
  python3 embed1d_algoA_demo.py --from-adj /mnt/.../adj_debug_once.txt
"""

import argparse
from collections import deque

# -------------------------------
# Algorithm A（論文の擬似コード）そのままの実装
# -------------------------------

def algoA_embed(root, children, is_leaf):
    """
    擬似コードに忠実に：
      f = 0 初期化
      treeEmbed(T, k=0) を呼び、葉だけ f[u] に 1D座標を設定して返す

    入力:
      root: ルートID
      children: List[List[int]] 子の隣接リスト（昇順推奨）
      is_leaf: List[int]（0/1）

    出力:
      f: dict {leaf_id -> x座標}
      H: 木の最大深さ
      depth: List[int] 各ノードの深さ
    """
    n = len(children)
    f = {u: 0 for u in range(n)}  # “f=0”の意味で全ノード0に初期化（葉にだけ最終値を残す）
    depth = [0]*n
    H = _compute_depths(root, children, depth)

    def add_on_leaves(u, delta):
        """部分木 u の葉にだけ f += delta"""
        stack = [u]
        while stack:
            x = stack.pop()
            if is_leaf[x]:
                f[x] += delta
            else:
                for v in children[x]:
                    stack.append(v)

    def treeEmbed(u, k):
        # if r is leaf then f(r)=k; return k
        if is_leaf[u]:
            f[u] = k
            return k
        # else: i=0, imax=(deg(r)-1), Δkold=0; for each child ~T:
        i = 0
        ch = children[u]
        imax = len(ch) - 1
        delta_old = 0
        for c in ch:
            kprime = treeEmbed(c, k)
            delta  = kprime - k
            k      = kprime
            if i > 0:
                gap = max(delta_old, delta)
                # foreach leaves x of ~T: f(x) += gap
                add_on_leaves(c, gap)
                k += gap
            if i != imax:
                k += 1
            delta_old = delta
            i += 1
        return k

    treeEmbed(root, 0)
    # root から到達できるノードだけに絞る
    reachable = [0]*n
    stack = [root]
    reachable[root] = 1
    while stack:
        u = stack.pop()
        for v in children[u]:
            if not reachable[v]:
                reachable[v] = 1
                stack.append(v)

    # 葉かつ到達可能だけ残す
    f = {u: f[u] for u in f if is_leaf[u] and reachable[u]}
    return f, H, depth



def _compute_depths(root, children, depth):
    """BFSで各ノード深さと最大深さを計算"""
    depth[root] = 0
    q = deque([root])
    H = 0
    while q:
        u = q.popleft()
        H = max(H, depth[u])
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)
    return H


def print_result(title, root, children, is_leaf, name, f, H, depth):
    print(title or "")
    print(f"max_depth(H)={H}, #leaves={sum(1 for x in is_leaf if x)}")
    print("葉(左→右): name, id, x, depth")
    # x座標でソート
    items = [(u, f[u]) for u in f]
    items.sort(key=lambda t: t[1])
    for u, x in items:
        nm = name[u] if 0 <= u < len(name) and name[u] else str(u)
        print(f" {nm:>3s}  (id={u:2d})  x={x:3d}  depth={depth[u]}")

def dump_dot(path, root, children, name=None, is_leaf=None, f=None, depth=None):
    n = len(children)

    # root から到達可能ノードだけ描く
    reachable = [0]*n
    stack = [root]
    reachable[root] = 1
    while stack:
        u = stack.pop()
        for v in children[u]:
            if not reachable[v]:
                reachable[v] = 1
                stack.append(v)

    def nm_of(u):
        if name and 0 <= u < len(name) and name[u]:
            return name[u]
        return str(u)

    def esc(s: str) -> str:
        # HTML-like label 用の最小エスケープ
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    def html_table_label(u):
        nm = esc(nm_of(u))
        d  = depth[u] if depth and 0 <= u < len(depth) else None
        x  = f.get(u) if (f is not None and u in f) else None

        # TABLE を使う（これが最も安定）
        rows = []
        rows.append(f"<TR><TD><B>{nm}</B></TD></TR>")
        rows.append(f"<TR><TD><FONT POINT-SIZE='9'>id={u}</FONT></TD></TR>")
        if d is not None:
            rows.append(f"<TR><TD><FONT POINT-SIZE='9'>d={d}</FONT></TD></TR>")
        if x is not None:
            # x=... だけ文字色を変える
            rows.append(f"<TR><TD><FONT POINT-SIZE='9' COLOR='blue'>x={x}</FONT></TD></TR>")

        table = "".join(rows)
        return f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>{table}</TABLE>>"

    with open(path, "w", encoding="utf-8") as g:
        g.write("digraph G {\n")
        g.write("  rankdir=TB;\n")
        g.write("  splines=true;\n")
        g.write("  nodesep=0.20;\n")
        g.write("  ranksep=0.35;\n")
        g.write("  edge [color=gray50 penwidth=1];\n")
        g.write("  node [fontsize=10 style=filled penwidth=1];\n")

        for u in range(n):
            if not reachable[u]:
                continue

            leaf = bool(is_leaf and is_leaf[u])

            if u == root:
                shape = "box"
                fill = "lightgray"
                color = "gray30"
            elif leaf:
                shape = "ellipse"
                fill = "lightyellow"   # 葉の塗り
                color = "orange"       # 葉の枠
            else:
                shape = "box"
                fill = "white"
                color = "gray40"

            label = html_table_label(u)
            g.write(f"  n{u} [shape={shape} fillcolor={fill} color={color} label={label}];\n")

        for u in range(n):
            if not reachable[u]:
                continue
            for v in children[u]:
                if reachable[v]:
                    g.write(f"  n{u} -> n{v};\n")

        g.write("}\n")


# -------------------------------
# 手作りツリー（実験モード）
# -------------------------------

def build_mode_A():
    """
    R-(A[a1,a2], B[b1,b2,b3], C[c1])
    R=0, A=1(a1=2,a2=3), B=4(b1=5,b2=6,b3=7), C=8(c1=9)
    """
    N = 10
    root = 0
    children = [[] for _ in range(N)]
    is_leaf = [0]*N
    name    = [""]*N

    def add(p,c): children[p].append(c)

    add(0,1); add(0,4); add(0,8)
    add(1,2); add(1,3)
    add(4,5); add(4,6); add(4,7)
    add(8,9)

    for i in range(N):
        children[i].sort()

    leaves = [2,3,5,6,7,9]
    for u in leaves: is_leaf[u]=1

    name[2]="a1"; name[3]="a2"
    name[5]="b1"; name[6]="b2"; name[7]="b3"
    name[9]="c1"
    name[0]="R"; name[1]="A"; name[4]="B"; name[8]="C"

    return root, children, is_leaf, name


def build_mode_Aprime():
    """
    R-(A[a1,a2], B[b1,b2,b3], C[leaf])
    C は葉（子なし）
    """
    N = 9
    root = 0
    children = [[] for _ in range(N)]
    is_leaf = [0]*N
    name    = [""]*N

    def add(p,c): children[p].append(c)

    add(0,1); add(0,4); add(0,8)
    add(1,2); add(1,3)
    add(4,5); add(4,6); add(4,7)

    for i in range(N): children[i].sort()
    leaves = [2,3,5,6,7,8]
    for u in leaves: is_leaf[u]=1

    name[2]="a1"; name[3]="a2"
    name[5]="b1"; name[6]="b2"; name[7]="b3"
    name[8]="C"
    name[0]="R"; name[1]="A"; name[4]="B"
    return root, children, is_leaf, name


def build_mode_C():
    """
    R-(X(A->a1, B->b1), Y(C->c1,c2))
    R=0; X=1(A=2->a1=3, B=4->b1=5); Y=6(C=7->c1=8,c2=9)
    """
    N=10
    root=0
    children=[[] for _ in range(N)]
    is_leaf=[0]*N
    name=[""]*N

    def add(p,c): children[p].append(c)

    add(0,1); add(0,6)
    add(1,2); add(2,3)  # A->a1
    add(1,4); add(4,5)  # B->b1
    add(6,7); add(7,8); add(7,9)  # C->(c1,c2)

    for i in range(N): children[i].sort()
    leaves=[3,5,8,9]
    for u in leaves: is_leaf[u]=1

    name[3]="a1"; name[5]="b1"; name[8]="c1"; name[9]="c2"
    name[0]="R"; name[1]="X"; name[2]="A"; name[4]="B"; name[6]="Y"; name[7]="C"
    return root, children, is_leaf, name


def build_mode_Cprime():
    """
    R-(X(A leaf, B leaf), Y(C->c1,c2))
    """
    N=10
    root=0
    children=[[] for _ in range(N)]
    is_leaf=[0]*N
    name=[""]*N

    def add(p,c): children[p].append(c)

    add(0,1); add(0,6)
    add(1,2); add(1,3)    # A, B を葉に
    add(6,7); add(7,8); add(7,9)

    for i in range(N): children[i].sort()
    leaves=[2,3,8,9]
    for u in leaves: is_leaf[u]=1

    name[2]="A"; name[3]="B"; name[8]="c1"; name[9]="c2"
    name[0]="R"; name[1]="X"; name[6]="Y"; name[7]="C"
    return root, children, is_leaf, name


def build_mode_Cdoubleprime():
    """
    R-(X(A->a1, B leaf), Y(C->c1,c2))
    """
    N=10
    root=0
    children=[[] for _ in range(N)]
    is_leaf=[0]*N
    name=[""]*N

    def add(p,c): children[p].append(c)

    add(0,1); add(0,6)
    add(1,2); add(2,3)  # A->a1
    add(1,4)            # B を葉に
    add(6,7); add(7,8); add(7,9)

    for i in range(N): children[i].sort()
    leaves=[3,4,8,9]
    for u in leaves: is_leaf[u]=1

    name[3]="a1"; name[4]="B"; name[8]="c1"; name[9]="c2"
    name[0]="R"; name[1]="X"; name[2]="A"; name[6]="Y"; name[7]="C"
    return root, children, is_leaf, name


def build_mode_D():
    """
    #1 (root)
    ├─ #0  [leaf: v47]
    ├─ #4
    │   └─ #3
    │       ├─ #2  [leaf: v83]
    │       └─ #12 [leaf: v10]
    ├─ #7
    │   └─ #6
    │       └─ #5  [leaf: v49]
    ├─ #8  [leaf: v38]
    ├─ #11
    │   └─ #10
    │       └─ #9  [leaf: v18]
    ├─ #15
    │   └─ #14
    │       └─ #13 [leaf: v0]
    ├─ #17
    │   └─ #16 [leaf: v86]
    ├─ #18 [leaf: v67]
    └─ #21
        └─ #20
            └─ #19 [leaf: v42]
    """
    N = 22
    children = [[] for _ in range(N)]
    is_leaf = [0]*N
    name = [f"#{i}" for i in range(N)]

    root = 1
    # 親子（EDGE）
    edges = [
        (1,0),(1,4),(1,7),(1,8),(1,11),(1,15),(1,17),(1,18),(1,21),
        (4,3),(3,2),(3,12),
        (7,6),(6,5),
        (11,10),(10,9),
        (15,14),(14,13),
        (17,16),
        (21,20),(20,19),
    ]
    for p,c in edges:
        children[p].append(c)

    # 葉フラグ（ISLEAF）
    leaf_set = {0,2,5,8,9,12,13,16,18,19}
    for u in leaf_set:
        is_leaf[u] = 1

    # 名前（葉は語彙ID表示に寄せる）
    vocab_of_leaf = {
        0: 47, 2: 83, 5: 49, 8: 38, 9: 18, 12: 10, 13: 0, 16: 86, 18: 67, 19: 42
    }
    name[root] = "#1(root)"
    for u,v in vocab_of_leaf.items():
        name[u] = f"v{v}"

    # 子は見やすさ目的で昇順並べ
    for ch in children:
        ch.sort()
    return root, children, is_leaf, name

def build_mode_Dprime_manual():
    """
    Mode D を“手動で”全葉の深さ=3に揃えるダミーパス追加版。
    もとの葉:
      depth=1: 0(v47), 8(v38), 18(v67) → +2 ノードずつ追加
      depth=2: 16(v86) → +1 ノード追加
      depth=3: 2(v83), 5(v49), 9(v18), 12(v10), 13(v0), 19(v42) → 変更なし
    追加ノードID: 22..28 を使用
    """
    N = 29
    children = [[] for _ in range(N)]
    is_leaf = [0]*N
    name = [f"#{i}" for i in range(N)]

    root = 1
    # 親子（EDGE）
    edges = [
        (1,0),(1,4),(1,7),(1,8),(1,11),(1,15),(1,17),(1,18),(1,21),
        (4,3),(3,2),(3,12),
        (7,6),(6,5),
        (11,10),(10,9),
        (15,14),(14,13),
        (17,16),
        (21,20),(20,19),
        (0,22),(22,23),   # v47 を depth=3 へ
        (8,24),(24,25),   # v38
        (18,26),(26,27),  # v67
        (16,28),          # v86 を depth=3 へ
    ]
    for p,c in edges:
        children[p].append(c)

    # 葉フラグ（ISLEAF）
    leaf_set = {23,2,5,25,9,12,13,28,27,19}
    for u in leaf_set:
        is_leaf[u] = 1

    # 名前（葉は語彙ID表示に寄せる）
    vocab_of_leaf = {
        23: 47, 2: 83, 5: 49, 25: 38, 9: 18, 12: 10, 13: 0, 28: 86, 27: 67, 19: 42
    }
    name[root] = "#1(root)"
    for u,v in vocab_of_leaf.items():
        name[u] = f"v{v}"

    # 子は見やすさ目的で昇順並べ
    for ch in children:
        ch.sort()
    return root, children, is_leaf, name

def build_mode_E2_packedstressplus():
    """
    Mode E2: PackedStressPlus
    - root has many children (12)
    - multiple big subtrees: BIG_L (16 leaves), BIG_M (12 leaves), BIG_R (16 leaves)
      including one in the "middle" (node 9)
    - plus medium subtrees, chains, shallow star, and a few root-direct leaves
    """
    N = 200
    root = 0
    children = [[] for _ in range(N)]
    is_leaf = [0]*N
    name = [f"#{i}" for i in range(N)]

    def add(p, c):
        children[p].append(c)

    # root children: 1..12
    for c in range(1, 13):
        add(0, c)

    next_id = 13

    def make_chain(parent, length, leaf_fanout):
        nonlocal next_id
        cur = parent
        for _ in range(length):
            v = next_id; next_id += 1
            add(cur, v)
            cur = v
        leaves = []
        for _ in range(leaf_fanout):
            lf = next_id; next_id += 1
            add(cur, lf)
            leaves.append(lf)
        return leaves

    def make_balanced(parent, depth, fanout):
        nonlocal next_id
        cur_level = [parent]
        for _ in range(depth):
            nxt = []
            for u in cur_level:
                for _ in range(fanout):
                    v = next_id; next_id += 1
                    add(u, v)
                    nxt.append(v)
            cur_level = nxt
        return cur_level  # leaves

    def make_mixed_big(parent, leaf_target, depth_main, fanout_main):
        leaves = make_balanced(parent, depth_main, fanout_main)
        while len(leaves) < leaf_target:
            leaves.extend(make_chain(parent, length=depth_main+1, leaf_fanout=2))
        return leaves[:leaf_target]

    # 1: BIG_L (16 leaves)
    name[1] = "BIG_L"
    bigL = make_mixed_big(1, leaf_target=16, depth_main=3, fanout_main=2)  # 8 + extra
    for i, lf in enumerate(bigL):
        name[lf] = f"L{i}"

    # 2: root-direct leaf
    name[2] = "t0"

    # 3: chain_small (3 internal, 3 leaves)
    name[3] = "C1"
    c1 = make_chain(3, length=3, leaf_fanout=3)
    for i, lf in enumerate(c1):
        name[lf] = f"c1_{i}"

    # 4: MED_1 (6 leaves)
    name[4] = "MED_1"
    med1 = make_balanced(4, depth=2, fanout=3)[:6]
    for i, lf in enumerate(med1):
        name[lf] = f"m1_{i}"

    # 5: root-direct leaf
    name[5] = "t1"

    # 6: chain_small (4 internal, 2 leaves)
    name[6] = "C2"
    c2 = make_chain(6, length=4, leaf_fanout=2)
    for i, lf in enumerate(c2):
        name[lf] = f"c2_{i}"

    # 7: shallow star (5 leaves)
    name[7] = "STAR"
    star = []
    for i in range(5):
        lf = next_id; next_id += 1
        add(7, lf)
        star.append(lf)
        name[lf] = f"s{i}"

    # 8: root-direct leaf
    name[8] = "t2"

    # 9: BIG_M (12 leaves)  <-- middle big subtree
    name[9] = "BIG_M"
    bigM = make_mixed_big(9, leaf_target=12, depth_main=3, fanout_main=2)
    for i, lf in enumerate(bigM):
        name[lf] = f"M{i}"

    # 10: MED_2 (8 leaves exactly)
    name[10] = "MED_2"
    med2 = make_balanced(10, depth=3, fanout=2)  # 8 leaves
    for i, lf in enumerate(med2):
        name[lf] = f"m2_{i}"

    # 11: chain_small (2 internal, 4 leaves)
    name[11] = "C3"
    c3 = make_chain(11, length=2, leaf_fanout=4)
    for i, lf in enumerate(c3):
        name[lf] = f"c3_{i}"

    # 12: BIG_R (16 leaves)
    name[12] = "BIG_R"
    bigR = make_mixed_big(12, leaf_target=16, depth_main=4, fanout_main=2)  # 16 leaves
    for i, lf in enumerate(bigR):
        name[lf] = f"R{i}"

    # deterministic child order
    for ch in children:
        ch.sort()

    # leaf flags: nodes with no children
    for u in range(N):
        if len(children[u]) == 0:
            is_leaf[u] = 1

    name[root] = "R"
    return root, children, is_leaf, name

# -------------------------------
# FlowTree サンドボックスの adj ダンプの読み込み
# -------------------------------

def load_adj_dump(path):
    """
    flowtree_sandbox --dump-adj が出すファイル:
      #PACKED_NODES N
      #ROOT r
      #EDGE
      p q
      p q
      ...
      #ISLEAF <N個の0/1>
      #UNLEAF <N個の整数(葉は語彙ID、それ以外は-1)>

    を読み、(root, children, is_leaf, name) を返す。
    名前は葉なら "v<UNLEAF[u]>"、それ以外は "#u" とする。
    """
    root = None
    N = None
    edges = []
    is_leaf = []
    unleaf  = []
    stage = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith("#PACKED_NODES"):
                N = int(s.split()[1])
            elif s.startswith("#ROOT"):
                root = int(s.split()[1])
            elif s.startswith("#EDGE"):
                stage = "EDGE"
            elif s.startswith("#ISLEAF"):
                toks = s.split()[1:]
                is_leaf = list(map(int, toks))
                stage = None
            elif s.startswith("#UNLEAF"):
                toks = s.split()[1:]
                unleaf = list(map(int, toks))
                stage = None
            elif s.startswith("#"):
                stage = None
            else:
                if stage == "EDGE":
                    p,q = s.split()
                    edges.append((int(p), int(q)))

    if N is None or root is None:
        raise ValueError("adj dump: PACKED_NODES/ROOT が不足しています")

    children = [[] for _ in range(N)]
    name = [f"#{i}" for i in range(N)]
    for p,q in edges:
        children[p].append(q)
    for ch in children:
        ch.sort()

    if not is_leaf:
        # 無ければ tout-tin==1 の代用…だが、adj には tin/tout が無いので
        # ここでは葉推定を簡略化:
        # 子を持たないノード＝葉として扱う
        is_leaf = [1 if len(children[u])==0 else 0 for u in range(N)]

    if unleaf and len(unleaf)==N:
        for u in range(N):
            if is_leaf[u] and unleaf[u] >= 0:
                name[u] = f"v{unleaf[u]}"

    return root, children, is_leaf, name


# -------------------------------
# メイン
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=[
        "A","Aprime","C","Cprime","Cdoubleprime","D","Dprime","Dprime_manual","E2"
    ])
    ap.add_argument("--from-adj", default="", help="flowtree_sandbox --dump-adj の出力ファイル")
    args = ap.parse_args()

    # 優先度：from-adj > mode
    if args.from_adj:
        root, children, is_leaf, name = load_adj_dump(args.from_adj)
        f, H, depth = algoA_embed(root, children, is_leaf)
        print(f"=== From Adj dump: {args.from_adj} ===")
        print_result("", root, children, is_leaf, name, f, H, depth)
        return

    if not args.mode:
        ap.error("モードか --from-adj を指定してください")

    if args.mode == "A":
        root, children, is_leaf, name = build_mode_A()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode A: R-(A[a1,a2], B[b1,b2,b3], C[c1]) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "Aprime":
        root, children, is_leaf, name = build_mode_Aprime()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode A': R-(A[a1,a2], B[b1,b2,b3], C[leaf]) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "C":
        root, children, is_leaf, name = build_mode_C()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode C: R-(X(A->a1, B->b1), Y(C->c1,c2)) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "Cprime":
        root, children, is_leaf, name = build_mode_Cprime()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode C': R-(X(A leaf, B leaf), Y(C->c1,c2)) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "Cdoubleprime":
        root, children, is_leaf, name = build_mode_Cdoubleprime()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode C'': R-(X(A->a1, B leaf), Y(C->c1,c2)) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "D":
        root, children, is_leaf, name = build_mode_D()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode D: packed-like hand-made tree ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return

    if args.mode == "Dprime_manual":
        root, children, is_leaf, name = build_mode_Dprime_manual()
        f, H, depth = algoA_embed(root, children, is_leaf)
        print("=== Mode D′ manual: D を手動ダミーパスで全葉深さ=3に揃えた版 ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return
    
    if args.mode == "E2":
        root, children, is_leaf, name = build_mode_E2_packedstressplus()
        f, H, depth = algoA_embed(root, children, is_leaf)
        dump_dot("modeE2.dot", root, children, name=name, is_leaf=is_leaf, f=f, depth=depth)
        print("=== Mode E2: packed-like stress test (many root children + multiple big subtrees) ===")
        print_result("", root, children, is_leaf, name, f, H, depth); return
    
        
if __name__ == "__main__":
    main()
