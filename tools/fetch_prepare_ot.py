# tools/fetch_prepare_ot.py
import os, re, sys, pathlib, argparse
import numpy as np
from tqdm import tqdm

# --- 英/日トークナイザ ---
_EN_STOP = set("a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves".split())
def tokenize_en(text):
    text = text.lower()
    toks = re.findall(r"[a-z]+", text)
    return [t for t in toks if t not in _EN_STOP and len(t) >= 2]

def tokenize_ja(text):
    from fugashi import Tagger
    if not hasattr(tokenize_ja, "_tagger"):
        tokenize_ja._tagger = Tagger()
    toks = []
    for w in tokenize_ja._tagger(text):
        base = getattr(w.feature, "lemma", None) or w.surface
        base = base.lower()
        if base and not re.fullmatch(r"[0-9０-９]+", base):
            toks.append(base)
    return toks

# --- 語彙＆埋め込み（gensim Word2Vecで自前学習） ---
def build_vocab_embeddings(tokenized_docs, vocab_size=20000, min_df=5, dim=100, epochs=5):
    from gensim.models import Word2Vec
    # 文書頻度で希少語を除外
    df = {}
    for t in tokenized_docs:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    docs = [[w for w in t if df[w] >= min_df] for t in tokenized_docs]
    # 学習
    model = Word2Vec(sentences=docs, vector_size=dim, window=5, min_count=min_df,
                     workers=max(1, os.cpu_count()-1), epochs=epochs)
    # 頻度順で語彙を切り出し
    from collections import Counter
    most = [w for w,_ in Counter([w for d in docs for w in d]).most_common(vocab_size*2)]
    keep = [w for w in most if w in model.wv][:vocab_size]
    word2id = {w:i for i,w in enumerate(keep)}
    vocab = np.vstack([model.wv[w] for w in keep]).astype(np.float32)
    return vocab, word2id

def docs_to_idlists(tokenized_docs, word2id, min_len=3, max_len=None):
    idlists=[]
    for toks in tokenized_docs:
        ids = sorted(set(word2id[w] for w in toks if w in word2id))
        if len(ids) >= min_len:
            idlists.append(ids[:max_len] if max_len else ids)
    return idlists

# --- データ取得 ---
def load_20news():
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all", remove=("headers","quotes","footers"), shuffle=True, random_state=0)
    return data.data  # list[str]

def load_hf(name):
    from datasets import load_dataset
    ds = load_dataset(name)  # ag_news / livedoor
    texts = []
    for split in ds.keys():
        col = "text" if "text" in ds[split].column_names else ds[split].column_names[0]
        texts += ds[split][col]
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["20news","ag_news","livedoor"], default="20news")
    ap.add_argument("--lang", choices=["en","ja","auto"], default="auto")
    ap.add_argument("--out_dir", default="./data/otdata")
    ap.add_argument("--vocab_size", type=int, default=20000)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()

    print(f"[1/5] fetching dataset: {args.source}")
    if args.source=="20news":
        texts = load_20news()
    elif args.source=="ag_news":
        texts = load_hf("ag_news")
    else:
        # livedoor ニュース（日本語）
        texts = load_hf("shunk031/livedoor-news-corpus")

    # 言語自動判定（ざっくり）
    lang = args.lang
    if lang=="auto":
        sample = "".join(texts[:50])
        lang = "ja" if re.search(r"[\u3000-\u30ff\u4e00-\u9fff]", sample) else "en"
    tok = tokenize_ja if lang=="ja" else tokenize_en

    print(f"[2/5] tokenizing ({lang}) ...")
    tokenized = [tok(x) for x in tqdm(texts, desc="tokenize")]
    tokenized = [t for t in tokenized if len(t)>=3]
    if len(tokenized) < 500:
        raise RuntimeError("Too few docs after tokenization. Try lowering --min_df or use another source.")

    print("[3/5] training embeddings (Word2Vec) ...")
    vocab, word2id = build_vocab_embeddings(tokenized, vocab_size=args.vocab_size,
                                            min_df=args.min_df, dim=args.dim, epochs=args.epochs)

    print("[4/5] building id lists ...")
    idlists = docs_to_idlists(tokenized, word2id, min_len=3)
    Q = min(args.queries, max(10, len(idlists)//10))
    queries = np.array(idlists[:Q], dtype=object)
    dataset = np.array(idlists[Q:], dtype=object)

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"vocab.npy", vocab)
    np.save(out/"dataset.npy", dataset, allow_pickle=True)
    np.save(out/"queries.npy", queries, allow_pickle=True)
    print(f"Saved to {out}  vocab:{vocab.shape}  dataset:{len(dataset)}  queries:{len(queries)}")

    print("[5/5] making answers.npy via FlowTree (Top-1) ...")
    # FlowTree で近似NNを作成（速い）。真のEMDにしたい場合は規模を落としてPOTで作り直す。
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"python"))
    import ot_estimators as ote
    def to_uniform(ids):
        w = 1.0/len(ids); return [(int(t), float(w)) for t in ids]
    dataset_w = [to_uniform(d) for d in dataset]
    queries_w = [to_uniform(q) for q in queries]
    solver = ote.OTEstimators()
    solver.load_vocabulary(vocab.astype(np.float32))
    solver.load_dataset(dataset_w)
    ids_all = np.arange(len(dataset), dtype=np.int32)
    ans = np.zeros((1, len(queries)), dtype=np.int32)
    buf_id = np.empty(1, dtype=np.int32); buf_sc = np.empty(1, dtype=np.float32)
    for i, q in enumerate(tqdm(queries_w, desc="flowtree NN")):
        solver.flowtree_rank(q, ids_all, buf_id, buf_sc, True)
        ans[0, i] = buf_id[0]
    np.save(out/"answers.npy", ans)
    print("answers.npy saved", ans.shape)

if __name__ == "__main__":
    main()
