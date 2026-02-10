// FlowTree sandbox CLI (+ Euler 1D embedding debug dump once -> now multi-dump):
//   data_dir/{vocab.npy, dataset.npz, query.npz, input_ids.npy} を読み、
//   FlowTree 距離で Top-k を標準出力 "id \t score" で返す。
//   --dump-euler を付けると、packed 部分木の Euler 順序と tin/tout/depth を
//   「最初の1回だけ」stderr に出力（以降はダンプしない）。
//   --dump-adj を付けると、packed 部分木の親子関係を adj 形式で
//   「複数回」(既定100回) data_dir/adj_dump.txt に追記で保存する。
//     - 1回ごとに #BEGIN_DUMP / #END_DUMP で区切り
//     - idx, doc_id, seed をメタ情報として残す
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
#include <limits>
#include <functional>

#include "third_party/cnpy/cnpy.h"

using Matrix  = std::vector<std::vector<float>>;              // (V, D)
using Measure = std::vector<std::pair<int32_t, float>>;       // [(id, weight)]

// 追加：親子ダンプの複数回出力制御
static std::string g_dump_adj_path;   // 空でなければダンプ先（追記）
static int g_dump_adj_limit = 0;      // 0なら無効、>0ならその回数まで
static int g_dump_adj_count = 0;      // すでに出した回数
static uint64_t g_run_seed = 0;       // 実行のseed（メタ用）

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

namespace ft {

static constexpr double EPS  = 1e-8;
static constexpr double EPS2 = 1e-5;

inline int32_t sign(float x) {
  if (std::fabs(x) < EPS) throw std::logic_error("computing sign of ~0");
  return (x > 0) ? 1 : -1;
}

struct FlowTree {
  // 入力
  const Matrix* dict = nullptr;

  // 語彙全体での木
  std::vector<int32_t> parents;  // node -> parent（rootは-1）
  std::vector<int32_t> leaf;     // vocab_id -> leaf_node_id

  // クエリごとの packed 木
  int32_t num_queries = 0;
  std::vector<int32_t> marked, node_id, id_node;
  std::vector<std::vector<int32_t>> subtree;
  std::vector<std::vector<std::pair<float,int32_t>>> excess;
  std::vector<float> delta_node;
  std::vector<int32_t> unleaf;

  // Euler / 1D embedding
  std::vector<int32_t> tin, tout, depth;
  std::vector<int32_t> euler, first_occ;
  int32_t timer = 0;

  // デバッグ出力フラグ
  bool dump_euler = false;

  float l2dist(int32_t u, int32_t v) const {
    const auto& a = (*dict)[u];
    const auto& b = (*dict)[v];
    float s = 0.0f;
    for (size_t j=0;j<a.size();++j) { float d = a[j]-b[j]; s += d*d; }
    return std::sqrt(s);
  }

  void build(const Matrix& vocab, uint64_t seed=0) {
    dict = &vocab;
    int32_t V=(int32_t)vocab.size(), D=(int32_t)vocab[0].size();
    float cmin=std::numeric_limits<float>::max(), cmax=std::numeric_limits<float>::lowest();
    for(int i=0;i<V;i++) for(int j=0;j<D;j++){ cmin=std::min(cmin,vocab[i][j]); cmax=std::max(cmax,vocab[i][j]); }
    float delta = cmax - cmin; cmin -= delta;

    std::mt19937_64 gen(seed?seed:std::random_device{}());
    std::uniform_real_distribution<float> ur(0.0f, delta);

    std::vector<std::pair<float,float>> box(D);
    for (int j=0;j<D;j++) { float s=ur(gen); box[j]={cmin+s, cmax+s}; }

    std::vector<int32_t> all(V); for (int i=0;i<V;i++) all[i]=i;

    parents.clear(); parents.reserve(2*V);
    leaf.assign(V, -1);

    build_quadtree(all, box, 0, -1, vocab);

    num_queries = 0;
    marked.assign(parents.size(), -1);
    node_id.assign(parents.size(), -1);
  }

  void build_quadtree(const std::vector<int32_t>& subset,
                      const std::vector<std::pair<float,float>>& box,
                      int32_t /*depth*/, int32_t parent_id,
                      const Matrix& vocab)
  {
    int32_t cur = (int32_t)parents.size();
    parents.push_back(parent_id);

    if ((int)subset.size() == 1) { leaf[subset[0]] = cur; return; }

    int32_t D=(int32_t)vocab[0].size();
    std::vector<float> mid(D);
    for (int32_t j=0;j<D;j++) mid[j] = (box[j].first + box[j].second)*0.5f;

    std::map<std::vector<uint8_t>, std::vector<int32_t>> parts;
    for (auto id : subset) {
      std::vector<uint8_t> code((D+7)/8, 0);
      for (int32_t j=0;j<D;j++) if (vocab[id][j] > mid[j]) code[j/8] |= (1u << (j%8));
      parts[code].push_back(id);
    }

    std::vector<std::pair<float,float>> new_box(D);
    for (const auto& kv : parts) {
      const auto& code = kv.first;
      for (int32_t j=0;j<D;j++) {
        uint8_t bit = (code[j/8] >> (j%8)) & 1u;
        new_box[j] = bit ? std::make_pair(mid[j], box[j].second)
                         : std::make_pair(box[j].first, mid[j]);
      }
      build_quadtree(kv.second, new_box, 0, cur, vocab);
    }
  }

  void build_euler_on_packed(int32_t root) {
    int32_t N = (int32_t)subtree.size();
    tin.assign(N, -1); tout.assign(N, -1); depth.assign(N, 0);
    first_occ.assign(N, -1);
    euler.clear(); euler.reserve(2*N);
    timer = 0;

    std::function<void(int32_t,int32_t)> dfs = [&](int32_t u, int32_t d){
      depth[u] = d;
      tin[u]   = timer++;                 // 入るとき
      if (first_occ[u] < 0) first_occ[u] = (int32_t)euler.size();
      euler.push_back(u);

      for (int32_t v : subtree[u]) {
        dfs(v, d+1);
        euler.push_back(u);               // 戻りがけ
      }
      tout[u] = timer;                    // 出るとき
    };
    dfs(root, 0);
  }

  // doc_id はダンプメタ用（main 側で呼び出し時に渡す）
  float flowtree_query(const Measure& a, const Measure& b, int32_t doc_id_for_dump = -1) {
    int32_t num_nodes = 0;
    id_node.clear();

    for (auto x : a) {
      int32_t id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) { id_node.push_back(id); node_id[id] = num_nodes++; }
        marked[id] = num_queries; id = parents[id];
      }
    }
    for (auto x : b) {
      int32_t id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) { id_node.push_back(id); node_id[id] = num_nodes++; }
        marked[id] = num_queries; id = parents[id];
      }
    }

    if ((int32_t)subtree.size() < num_nodes) subtree.resize(num_nodes);
    for (int32_t i=0;i<num_nodes;i++) subtree[i].clear();
    for (int32_t i=0;i<num_nodes;i++) {
      int32_t u = parents[id_node[i]];
      if (u != -1 && node_id[u] != -1) subtree[node_id[u]].push_back(i);
    }
    if ((int32_t)excess.size() < num_nodes) excess.resize(num_nodes);
    delta_node.assign(num_nodes, 0.0f);
    unleaf.resize(num_nodes);

    for (auto x : a) { int32_t nd = node_id[leaf[x.first]]; delta_node[nd] += x.second; unleaf[nd] = x.first; }
    for (auto x : b) { int32_t nd = node_id[leaf[x.first]]; delta_node[nd] -= x.second; unleaf[nd] = x.first; }

    int32_t root = node_id[0];
    build_euler_on_packed(root);

    // 追加：親子ダンプ（複数回 / 追記）
    if (!g_dump_adj_path.empty() && g_dump_adj_limit > 0 && g_dump_adj_count < g_dump_adj_limit) {
      std::ofstream ofs(g_dump_adj_path, std::ios::app);
      if (!ofs) throw std::runtime_error("failed to open dump-adj output for append: " + g_dump_adj_path);

      int32_t N = (int32_t)subtree.size();

      // ブロック開始
      ofs << "#BEGIN_DUMP"
          << " idx=" << g_dump_adj_count
          << " doc_id=" << doc_id_for_dump
          << " seed=" << (unsigned long long)g_run_seed
          << "\n";

      ofs << "#PACKED_NODES " << N << "\n";
      ofs << "#ROOT " << root << "\n";

      // 親子エッジ
      ofs << "#EDGE\n";
      int64_t edge_count = 0;
      for (int u = 0; u < N; ++u) {
        for (int v : subtree[u]) { ofs << u << " " << v << "\n"; edge_count++; }
      }

      // 葉フラグ
      ofs << "#ISLEAF";
      for (int u = 0; u < N; ++u) {
        int isleaf = subtree[u].empty() ? 1 : 0;
        ofs << " " << isleaf;
      }
      ofs << "\n";

      // 語彙ID（葉のみ有効、内部は -1）
      ofs << "#UNLEAF";
      for (int u = 0; u < N; ++u) {
        if (subtree[u].empty()) ofs << " " << unleaf[u];
        else ofs << " " << -1;
      }
      ofs << "\n";

      // ブロック終了（簡易整合用情報も添える）
      ofs << "#END_DUMP"
          << " idx=" << g_dump_adj_count
          << " edge_count=" << edge_count
          << "\n";
      ofs.close();

      g_dump_adj_count++;
    }

    if (dump_euler) {
      std::cerr << "#PACKED_NODES " << num_nodes << "\n";
      std::cerr << "#ROOT " << root << "\n";
      std::cerr << "#TIN";   for (int i=0;i<num_nodes;i++) std::cerr << " " << tin[i];   std::cerr << "\n";
      std::cerr << "#TOUT";  for (int i=0;i<num_nodes;i++) std::cerr << " " << tout[i];  std::cerr << "\n";
      std::cerr << "#DEPTH"; for (int i=0;i<num_nodes;i++) std::cerr << " " << depth[i]; std::cerr << "\n";
      std::cerr << "#FIRST"; for (int i=0;i<num_nodes;i++) std::cerr << " " << first_occ[i]; std::cerr << "\n";
      std::cerr << "#EULER"; for (size_t i=0;i<euler.size();i++) std::cerr << " " << euler[i]; std::cerr << "\n";
      std::cerr << "#UNLEAF"; for (int i=0;i<num_nodes;i++) std::cerr << " " << unleaf[i]; std::cerr << "\n";
    }

    float res = run_query(0, root);

    if (!excess[root].empty()) {
      float unassigned = 0.0f; for (auto &p : excess[root]) unassigned += p.first;
      if (unassigned > EPS2) throw std::logic_error("too much unassigned flow");
    }
    ++num_queries;
    return res;
  }

  float run_query(int32_t /*depth*/, int32_t nd) {
    float res = 0.0f;
    for (auto ch : subtree[nd]) res += run_query(0, ch);
    excess[nd].clear();

    if (subtree[nd].empty()) {
      if (std::fabs(delta_node[nd]) > EPS) excess[nd].push_back({delta_node[nd], unleaf[nd]});
      return res;
    }

    for (auto ch : subtree[nd]) {
      if (excess[ch].empty()) continue;
      bool same=false;
      if (excess[nd].empty()) same=true;
      else if (sign(excess[ch].back().first) == sign(excess[nd].back().first)) same=true;

      if (same) {
        for (auto &t : excess[ch]) excess[nd].push_back(t);
      } else {
        auto &A = excess[nd], &B = excess[ch];
        while (!A.empty() && !B.empty()) {
          auto u = A.back(), v = B.back();
          float dist = l2dist(u.second, v.second);
          if (std::fabs(u.first + v.first) < EPS) { A.pop_back(); B.pop_back(); res += dist * std::fabs(u.first); }
          else if (std::fabs(u.first) < std::fabs(v.first)) { A.pop_back(); B.back().first += u.first; res += dist * std::fabs(u.first); }
          else { B.pop_back(); A.back().first += v.first; res += dist * std::fabs(v.first); }
        }
        if (!B.empty()) B.swap(A);
      }
    }
    return res;
  }
};

} // namespace ft

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <data_dir> <topk> [seed] [--dump-euler] [--dump-adj] [--dump-adj-limit N] [--dump-adj-out FILE]\n"
                 "  expects: data_dir/{vocab.npy,dataset.npz,query.npz,input_ids.npy}\n";
    return 1;
  }

  std::string data_dir = argv[1];
  int topk = std::stoi(argv[2]);
  uint64_t seed = 0;
  bool want_dump_euler = false;

  // 追加:
  bool want_dump_adj = false;
  int dump_adj_limit = 100;                 // 既定100回
  std::string dump_adj_out = data_dir + "/adj_dump.txt"; // 既定出力先

  // 引数パース（順不同）
  for (int i=3; i<argc; ++i) {
    std::string tok = argv[i];
    if (tok == "--dump-euler") {
      want_dump_euler = true;
    } else if (tok == "--dump-adj") {
      want_dump_adj = true;
    } else if (tok == "--dump-adj-limit") {
      if (i+1 >= argc) { std::cerr << "--dump-adj-limit needs an integer\n"; return 1; }
      dump_adj_limit = std::stoi(argv[++i]);
    } else if (tok == "--dump-adj-out") {
      if (i+1 >= argc) { std::cerr << "--dump-adj-out needs a filename\n"; return 1; }
      dump_adj_out = argv[++i];
    } else {
      // seed かもしれない
      try { seed = std::stoull(tok); } catch (...) { /* ignore */ }
    }
  }

  // グローバル：ダンプ設定
  if (want_dump_adj) {
    g_dump_adj_path  = dump_adj_out;
    g_dump_adj_limit = std::max(0, dump_adj_limit);
    g_dump_adj_count = 0;
    g_run_seed       = seed;
    // 既存ファイルがあれば追記になるので、毎回新しく取りたいなら最初に消す
    // ここでは「この実行のログ」を綺麗にするため、先頭でtruncateする（推奨）
    {
      std::ofstream ofs(g_dump_adj_path, std::ios::trunc);
      if (!ofs) { std::cerr << "failed to open dump file: " << g_dump_adj_path << "\n"; return 1; }
      ofs << "#FLOWTREE_ADJ_DUMPS"
          << " seed=" << (unsigned long long)seed
          << " limit=" << g_dump_adj_limit
          << "\n";
    }
  } else {
    g_dump_adj_path.clear();
    g_dump_adj_limit = 0;
    g_dump_adj_count = 0;
    g_run_seed = seed;
  }

  auto path = [&](const char* f){ return data_dir + "/" + f; };

  Matrix vocab = load_vocab_npy(path("vocab.npy"));
  auto dataset = load_dataset_npz(path("dataset.npz"));
  Measure query = load_query_npz(path("query.npz"));
  auto input_ids = load_input_ids_npy(path("input_ids.npy"));
  if (topk > (int)input_ids.size()) topk = (int)input_ids.size();

  ft::FlowTree tree;
  tree.build(vocab, seed);

  bool first_euler_done = false;  // 最初の1回だけ euler を出す

  std::vector<std::pair<float,int32_t>> dist; dist.reserve(input_ids.size());
  for (int32_t id : input_ids) {
    // euler は最初の1回だけ
    tree.dump_euler = (want_dump_euler && !first_euler_done);

    // packed 木ダンプは flowtree_query 内で g_dump_adj_count < limit の間だけ出る
    float score = tree.flowtree_query(query, dataset[id], /*doc_id_for_dump=*/id);
    dist.emplace_back(score, id);

    if (want_dump_euler && !first_euler_done) first_euler_done = true;

    // すでに必要な回数の adj を出し切ったなら、残りはランキングだけ継続（通常）
    // もし「adj を出し切ったらランキングも止めたい」なら break してOK。
  }

  std::nth_element(dist.begin(), dist.begin()+topk, dist.end());
  std::sort(dist.begin(), dist.begin()+topk);
  for (int i=0;i<topk;++i) std::cout << dist[i].second << "\t" << dist[i].first << "\n";
  return 0;
}
