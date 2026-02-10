// FlowTree sandbox CLI:
//   data_dir/{vocab.npy, dataset.npz, query.npz, input_ids.npy} を読み、
//   FlowTree 距離で Top-k を標準出力に "id \t score" で返す。
// 依存: cnpy（third_party/cnpy/cnpy.{h,cpp}）, zlib
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "third_party/cnpy/cnpy.h"

// ------------------ 型 ------------------
using Matrix  = std::vector<std::vector<float>>;              // (V, D)
using Measure = std::vector<std::pair<int32_t, float>>;       // [(id, weight)]

// ------------------ npy/npz ロード ------------------
static Matrix load_vocab_npy(const std::string& path) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  if (arr.word_size != 4 || arr.shape.size() != 2) {
    throw std::runtime_error("vocab.npy must be float32 2D");
  }
  size_t V = arr.shape[0], D = arr.shape[1];
  const float* p = arr.data<float>();
  Matrix m(V, std::vector<float>(D));
  for (size_t i=0;i<V;i++) for (size_t j=0;j<D;j++) m[i][j] = p[i*D+j];
  return m;
}

static std::vector<Measure> load_dataset_npz(const std::string& path) {
  auto npz = cnpy::npz_load(path);
  auto ids     = npz.at("ids").as_vec<int32_t>();
  auto weights = npz.at("weights").as_vec<float>();
  auto offs    = npz.at("offsets").as_vec<int32_t>();
  if (offs.empty()) throw std::runtime_error("offsets empty");
  std::vector<Measure> ds; ds.reserve(offs.size()-1);
  for (size_t k=0; k+1<offs.size(); ++k) {
    int32_t s = offs[k], e = offs[k+1];
    Measure m; m.reserve(e-s);
    for (int i=s;i<e;++i) m.emplace_back(ids[i], weights[i]);
    ds.push_back(std::move(m));
  }
  return ds;
}

static Measure load_query_npz(const std::string& path) {
  auto npz = cnpy::npz_load(path);
  auto ids = npz.at("ids").as_vec<int32_t>();
  auto ws  = npz.at("weights").as_vec<float>();
  Measure q; q.reserve(ids.size());
  for (size_t i=0;i<ids.size();++i) q.emplace_back(ids[i], ws[i]);
  return q;
}

static std::vector<int32_t> load_input_ids_npy(const std::string& path) {
  return cnpy::npy_load(path).as_vec<int32_t>();
}

// ------------------ FlowTree 本体 ------------------
namespace ft {

static constexpr double EPS  = 1e-8;
static constexpr double EPS2 = 1e-5;

inline int32_t sign(float x) {
  if (std::fabs(x) < EPS) {
    throw std::logic_error("computing sign of ~0");
  }
  return (x > 0) ? 1 : -1;
}

struct FlowTree {
  // 入力
  const Matrix* dict;                 // 語彙ベクトル (V,D)

  // 木構造
  std::vector<int32_t> parents;       // node -> parent (root の parent = -1)
  std::vector<int32_t> leaf;          // vocab_id -> leaf_node_id

  // クエリ一時領域
  int32_t num_queries = 0;
  std::vector<int32_t> marked;        // node が今回のクエリで訪問済みか
  std::vector<int32_t> node_id;       // original node id -> packed id
  std::vector<int32_t> id_node;       // packed id -> original node id
  std::vector<std::vector<int32_t>> subtree; // packed 木の子リスト
  std::vector<std::vector<std::pair<float,int32_t>>> excess; // 余剰 (mass, vocab_id)
  std::vector<float> delta_node;      // packed node ごとの +/− 質量合計
  std::vector<int32_t> unleaf;        // packed leaf に対応する語彙 id（デバッグ/距離用）

  FlowTree() : dict(nullptr) {}

  // L2距離
  float l2dist(int32_t u, int32_t v) const {
    const auto& a = (*dict)[u];
    const auto& b = (*dict)[v];
    float s = 0.0f;
    for (size_t j=0;j<a.size();++j) {
      float d = a[j]-b[j]; s += d*d;
    }
    return std::sqrt(s);
  }

  // 語彙から木を構築（ランダムシフトあり）
  void build(const Matrix& vocab, uint64_t seed = 0) {
    dict = &vocab;
    const int32_t V = (int32_t)vocab.size();
    const int32_t D = (int32_t)vocab[0].size();

    // cmin/cmax（全次元全語彙）
    float cmin = std::numeric_limits<float>::max();
    float cmax = std::numeric_limits<float>::lowest();
    for (int i=0;i<V;i++) for (int j=0;j<D;j++) {
      cmin = std::min(cmin, vocab[i][j]);
      cmax = std::max(cmax, vocab[i][j]);
    }
    float delta = cmax - cmin;
    cmin -= delta; // 元実装準拠

    // ランダムシフト ∈ [0, delta]
    std::mt19937_64 gen(seed ? seed : std::random_device{}());
    std::uniform_real_distribution<float> ur(0.0f, delta);

    std::vector<std::pair<float,float>> box(D);
    for (int j=0;j<D;j++) {
      float s = ur(gen);
      box[j] = std::make_pair(cmin + s, cmax + s);
    }

    // 全語彙ID
    std::vector<int32_t> all(V);
    for (int32_t i=0;i<V;i++) all[i] = i;

    // 初期化
    parents.clear(); parents.reserve(2*V);
    leaf.assign(V, -1);

    // 再帰でクワッドツリー構築
    build_quadtree(all, box, /*depth=*/0, /*parent=*/-1, vocab);

    // クエリ用バッファ
    num_queries = 0;
    marked.assign(parents.size(), -1);
    node_id.assign(parents.size(), -1);
  }

  // クワッドツリー構築（元実装に準拠）
  void build_quadtree(const std::vector<int32_t>& subset,
                      const std::vector<std::pair<float,float>>& box,
                      int32_t /*depth*/, int32_t parent_id,
                      const Matrix& vocab)
  {
    int32_t cur = (int32_t)parents.size();
    parents.push_back(parent_id);

    if ((int)subset.size() == 1) {
      leaf[subset[0]] = cur;
      return;
    }

    const int32_t D = (int32_t)vocab[0].size();
    std::vector<float> mid(D);
    for (int32_t j=0;j<D;j++) {
      mid[j] = (box[j].first + box[j].second) * 0.5f;
    }

    // 分割：コード（d ビット）を 8bit パックした vector<uint8_t> を key にする
    std::map<std::vector<uint8_t>, std::vector<int32_t>> parts;
    for (auto id : subset) {
      std::vector<uint8_t> code((D + 7)/8, 0);
      for (int32_t j=0;j<D;j++) {
        if (vocab[id][j] > mid[j]) code[j/8] |= (1u << (j%8));
      }
      parts[code].push_back(id);
    }

    // 各子へ再帰
    std::vector<std::pair<float,float>> new_box(D);
    for (const auto& kv : parts) {
      const auto& code = kv.first;
      for (int32_t j=0;j<D;j++) {
        uint8_t bit = (code[j/8] >> (j%8)) & 1u;
        if (bit) new_box[j] = std::make_pair(mid[j], box[j].second);
        else     new_box[j] = std::make_pair(box[j].first, mid[j]);
      }
      build_quadtree(kv.second, new_box, /*depth+1*/0, cur, vocab);
    }
  }

  // FlowTree 距離（2測度 a,b）
  float flowtree_query(const Measure& a, const Measure& b) {
    int32_t num_nodes = 0;
    id_node.clear();

    // a 側の祖先をマーキング
    for (auto x : a) {
      int32_t id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }
    // b 側の祖先も
    for (auto x : b) {
      int32_t id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }

    // packed 木を作る
    if ((int32_t)subtree.size() < num_nodes) subtree.resize(num_nodes);
    for (int32_t i=0;i<num_nodes;i++) subtree[i].clear();
    for (int32_t i=0;i<num_nodes;i++) {
      int32_t u = parents[id_node[i]];
      if (u != -1 && node_id[u] != -1) {
        subtree[node_id[u]].push_back(i);
      }
    }
    if ((int32_t)excess.size() < num_nodes) excess.resize(num_nodes);
    delta_node.assign(num_nodes, 0.0f);
    unleaf.resize(num_nodes);

    // 葉に +/− 質量を加算
    for (auto x : a) {
      int32_t nd = node_id[leaf[x.first]];
      delta_node[nd] += x.second;
      unleaf[nd] = x.first;
    }
    for (auto x : b) {
      int32_t nd = node_id[leaf[x.first]];
      delta_node[nd] -= x.second;
      unleaf[nd] = x.first;
    }

    // root は original node id = 0 として良い（最初に push_back した）
    float res = run_query(/*depth=*/0, node_id[0]);

    // 余剰チェック
    if (!excess[node_id[0]].empty()) {
      float unassigned = 0.0f;
      for (auto &p : excess[node_id[0]]) unassigned += p.first;
      if (unassigned > EPS2) {
        throw std::logic_error("too much unassigned flow");
      }
    }
    ++num_queries;
    return res;
  }

  // 再帰：子から余剰を集め、末尾どうしで相殺
  float run_query(int32_t /*depth*/, int32_t nd) {
    float res = 0.0f;
    for (auto ch : subtree[nd]) {
      res += run_query(/*depth+1*/0, ch);
    }
    excess[nd].clear();

    if (subtree[nd].empty()) {
      if (std::fabs(delta_node[nd]) > EPS) {
        excess[nd].push_back({delta_node[nd], unleaf[nd]});
      }
      return res;
    }

    for (auto ch : subtree[nd]) {
      if (excess[ch].empty()) continue;

      bool same = false;
      if (excess[nd].empty()) same = true;
      else if (sign(excess[ch].back().first) == sign(excess[nd].back().first))
        same = true;

      if (same) {
        // 符号が同じ → そのまま積む
        for (auto &t : excess[ch]) excess[nd].push_back(t);
      } else {
        // 異符号 → 末尾同士で相殺
        auto &A = excess[nd];
        auto &B = excess[ch];
        while (!A.empty() && !B.empty()) {
          auto u = A.back();
          auto v = B.back();
          float dist = l2dist(u.second, v.second);

          if (std::fabs(u.first + v.first) < EPS) {
            A.pop_back(); B.pop_back();
            res += dist * std::fabs(u.first);
          } else if (std::fabs(u.first) < std::fabs(v.first)) {
            A.pop_back();
            B.back().first += u.first; // v の量を減らす（符号に注意：u.first は負の可能性あり）
            res += dist * std::fabs(u.first);
          } else {
            B.pop_back();
            A.back().first += v.first;
            res += dist * std::fabs(v.first);
          }
        }
        if (!B.empty()) B.swap(A); // B側に残っていたら nd 側へ持ってくる
      }
    }
    return res;
  }
};

} // namespace ft

// ------------------ main ------------------
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <data_dir> <topk> [seed]\n"
                 "  expects: data_dir/{vocab.npy,dataset.npz,query.npz,input_ids.npy}\n";
    return 1;
  }
  std::string data_dir = argv[1];
  int topk = std::stoi(argv[2]);
  uint64_t seed = 0;
  if (argc >= 4) {
    try { seed = std::stoull(argv[3]); } catch (...) { seed = 0; }
  }

  auto path = [&](const char* f){ return data_dir + "/" + f; };

  // 読み込み
  Matrix vocab = load_vocab_npy(path("vocab.npy"));
  auto dataset = load_dataset_npz(path("dataset.npz"));
  Measure query = load_query_npz(path("query.npz"));
  auto input_ids = load_input_ids_npy(path("input_ids.npy"));
  if (topk > (int)input_ids.size()) topk = (int)input_ids.size();

  // FlowTree 構築（語彙全体に対して一度だけ）
  ft::FlowTree tree;
  tree.build(vocab, seed); // ランダムシフトの seed を指定可能

  // 全候補に対して FlowTree 距離
  std::vector<std::pair<float,int32_t>> dist; dist.reserve(input_ids.size());
  for (int32_t id : input_ids) {
    float score = tree.flowtree_query(query, dataset[id]);
    dist.emplace_back(score, id);
  }

  // Top-k
  std::nth_element(dist.begin(), dist.begin()+topk, dist.end());
  std::sort(dist.begin(), dist.begin()+topk);
  for (int i=0;i<topk;++i) {
    std::cout << dist[i].second << "\t" << dist[i].first << "\n";
  }
  return 0;
}
