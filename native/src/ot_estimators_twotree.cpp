#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ote {

const double EPS = 1e-8;
const double EPS2 = 1e-5;
const double EPS3 = 1e-3;

inline int32_t sign(float x) {
  if (fabs(x) < EPS) {
    throw std::logic_error("computing sign of ~0");
  }
  if (x > 0) return 1;
  return -1;
}

class OTEstimators {
 public:
  using NumPyFloatArray = py::array_t<float, py::array::c_style>;
  using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

  using EigenVector = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Matrix = Eigen::Map<EigenMatrix>;

  OTEstimators() : stage(0), num_queries(0) {}

  // [MOD] seed 指定で木構築時のランダムシフトを再現可能にする。
  // [MOD] seed < 0 のときだけ従来どおり random_device を使う。
  void load_vocabulary(NumPyFloatArray points, int64_t seed = -1) {
    if (stage != 0) {
      throw std::logic_error(
          "load_vocabulary() should be called once in the beginning");
    }
    stage = 1;
    py::buffer_info buf = points.request();
    if (buf.ndim != 2) {
      throw std::logic_error(
          "load_vocabulary() expects a two-dimensional NumPy array");
    }
    auto n = buf.shape[0];
    auto d = buf.shape[1];
    dictionary = std::make_unique<Matrix>(static_cast<float *>(buf.ptr), n, d);

    auto cmin = std::numeric_limits<float>::max();
    auto cmax = std::numeric_limits<float>::lowest();
    for (ssize_t i = 0; i < n; ++i) {
      for (ssize_t j = 0; j < d; ++j) {
        cmin = std::min(cmin, (*dictionary)(i, j));
        cmax = std::max(cmax, (*dictionary)(i, j));
      }
    }

    auto delta = cmax - cmin;
    cmin -= delta;


    // [MOD] seed が与えられた場合はその値で乱数生成器を初期化する。
    // [MOD] これにより、異なる seed で異なる木を再現可能にする。
    std::mt19937_64 gen;
    if (seed >= 0) {
        gen.seed(static_cast<uint64_t>(seed));
    } else {
        std::random_device rd;
        uint64_t used_seed =
          (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        gen.seed(used_seed);
    }
    std::uniform_real_distribution<float> shift_gen(0.0f, delta);

    std::vector<std::pair<float, float>> bounding_box;
    for (ssize_t i = 0; i < d; ++i) {
      auto s = shift_gen(gen);
      bounding_box.push_back(std::make_pair(cmin + s, cmax + s));
    }

    std::vector<int32_t> all;
    for (ssize_t i = 0; i < n; ++i) {
      all.push_back(static_cast<int32_t>(i));
    }

    leaf.clear();
    leaf.resize(n);
    parents.clear();

    build_quadtree(all, bounding_box, 0, -1);

    num_queries = 0;
    marked.resize(parents.size());
    for (auto &x : marked) {
      x = -1;
    }
    node_id.resize(parents.size());
  }

  void load_dataset(
      const std::vector<std::vector<std::pair<int32_t, float>>> &dataset) {
    if (stage != 1) {
      throw std::logic_error(
          "load_dataset() should be called once after calling "
          "load_vocabulary()");
    }
    stage = 2;
    if (dataset.empty()) {
      throw std::logic_error("the dataset can't be empty");
    }

    dataset_embedding.clear();
    raw_dataset.clear();

    for (auto &measure : dataset) {
      check_measure(measure);
      dataset_embedding.push_back(compute_embedding(measure));
    }

    raw_dataset = dataset;
    for (auto &measure : raw_dataset) {
      std::sort(measure.begin(), measure.end());
    }

    means.resize(dataset.size(), dictionary->cols());
    for (size_t i = 0; i < dataset.size(); ++i) {
      means.row(i) =
          dictionary->row(dataset[i][0].first) * dataset[i][0].second;
      for (size_t j = 1; j < dataset[i].size(); ++j) {
        means.row(i) +=
            dictionary->row(dataset[i][j].first) * dataset[i][j].second;
      }
    }
    query_mean.resize(dictionary->cols());
    distances.resize(dataset.size());
  }

  void means_rank(const std::vector<std::pair<int32_t, float>> &query,
                  NumPyIntArray input_ids, NumPyIntArray output_ids,
                  NumPyFloatArray output_scores, bool to_sort) {
    check_stage();
    check_measure(query);
    check_input_output_arrays(input_ids, output_ids, output_scores);

    query_mean = dictionary->row(query[0].first) * query[0].second;
    for (size_t j = 1; j < query.size(); ++j) {
      query_mean =
          query_mean + dictionary->row(query[j].first) * query[j].second;
    }

    auto input_ids_buf = input_ids.request();
    auto input_ids_raw = static_cast<int32_t *>(input_ids_buf.ptr);
    int32_t k1 = static_cast<int32_t>(input_ids_buf.shape[0]);

    for (int32_t i = 0; i < k1; ++i) {
      float score = (query_mean - means.row(input_ids_raw[i])).squaredNorm();
      distances[i] = std::make_pair(score, input_ids_raw[i]);
    }
    select_topk_aux(k1, output_ids, output_scores, to_sort);
  }

  void overlap_rank(const std::vector<std::pair<int32_t, float>> &query,
                    NumPyIntArray input_ids, NumPyIntArray output_ids,
                    NumPyFloatArray output_scores, bool to_sort) {
    check_stage();
    check_measure(query);

    auto query_copy = query;
    std::sort(query_copy.begin(), query_copy.end());

    auto input_ids_buf = input_ids.request();
    auto input_ids_raw = static_cast<int32_t *>(input_ids_buf.ptr);
    int32_t k1 = static_cast<int32_t>(input_ids_buf.shape[0]);

    for (int32_t i = 0; i < k1; ++i) {
      auto &point = raw_dataset[input_ids_raw[i]];
      float score = 0.0f;
      size_t qp = 0;
      size_t dp = 0;

      while (qp < query_copy.size() && dp < point.size()) {
        if (query_copy[qp].first < point[dp].first) {
          ++qp;
        } else if (query_copy[qp].first > point[dp].first) {
          ++dp;
        } else {
          score += 1.0f;
          ++dp;
          ++qp;
        }
      }
      distances[i] = std::make_pair(-score, input_ids_raw[i]);
    }
    select_topk_aux(k1, output_ids, output_scores, to_sort);
  }

  void quadtree_rank(const std::vector<std::pair<int32_t, float>> &query,
                     NumPyIntArray input_ids, NumPyIntArray output_ids,
                     NumPyFloatArray output_scores, bool to_sort) {
    check_stage();
    check_measure(query);
    check_input_output_arrays(input_ids, output_ids, output_scores);

    auto query_embedding = compute_embedding(query);

    auto input_ids_buf = input_ids.request();
    auto input_ids_raw = static_cast<int32_t *>(input_ids_buf.ptr);
    int32_t k1 = static_cast<int32_t>(input_ids_buf.shape[0]);

    for (int32_t i = 0; i < k1; ++i) {
      auto &point_embedding = dataset_embedding[input_ids_raw[i]];
      float score = 0.0f;
      size_t qp = 0;
      size_t dp = 0;
      while (qp < query_embedding.size() || dp < point_embedding.size()) {
        if (qp == query_embedding.size()) {
          score += point_embedding[dp].second;
          ++dp;
        } else if (dp == point_embedding.size()) {
          score += query_embedding[qp].second;
          ++qp;
        } else if (query_embedding[qp].first < point_embedding[dp].first) {
          score += query_embedding[qp].second;
          ++qp;
        } else if (point_embedding[dp].first < query_embedding[qp].first) {
          score += point_embedding[dp].second;
          ++dp;
        } else {
          score += fabs(query_embedding[qp].second - point_embedding[dp].second);
          ++qp;
          ++dp;
        }
      }
      distances[i] = std::make_pair(score, input_ids_raw[i]);
    }
    select_topk_aux(k1, output_ids, output_scores, to_sort);
  }

  void flowtree_rank(const std::vector<std::pair<int32_t, float>> &query,
                     NumPyIntArray input_ids, NumPyIntArray output_ids,
                     NumPyFloatArray output_scores, bool to_sort) {
    check_stage();
    check_measure(query);
    check_input_output_arrays(input_ids, output_ids, output_scores);

    auto input_ids_buf = input_ids.request();
    auto input_ids_raw = static_cast<int32_t *>(input_ids_buf.ptr);
    int32_t k1 = static_cast<int32_t>(input_ids_buf.shape[0]);

    for (int32_t i = 0; i < k1; ++i) {
      auto cur_id = input_ids_raw[i];
      auto score = flowtree_query(query, raw_dataset[cur_id]);
      distances[i] = std::make_pair(score, cur_id);
    }
    select_topk_aux(k1, output_ids, output_scores, to_sort);
  }

  void select_topk(NumPyIntArray input_ids, NumPyFloatArray input_scores,
                   NumPyIntArray output_ids, NumPyFloatArray output_scores,
                   bool to_sort) {
    check_input_output_arrays(input_ids, input_scores, output_ids,
                              output_scores);

    auto input_scores_buf = input_scores.request();
    auto input_ids_buf = input_ids.request();
    auto input_scores_raw = static_cast<float *>(input_scores_buf.ptr);
    auto input_ids_raw = static_cast<int32_t *>(input_ids_buf.ptr);
    int32_t k1 = static_cast<int32_t>(input_ids_buf.shape[0]);

    for (int32_t i = 0; i < k1; ++i) {
      distances[i] = std::make_pair(input_scores_raw[i], input_ids_raw[i]);
    }
    select_topk_aux(k1, output_ids, output_scores, to_sort);
  }

  // [MOD] query-doc の support に必要な祖先だけを集め、
  // [MOD] packed subtree(local tree) と葉差分質量 delta_node を構築する。
  // [MOD] 元の flowtree_query(...) が持っていた前半の責務を切り出した。
  void build_packed_subtree(
      const std::vector<std::pair<int32_t, float>> &a,
      const std::vector<std::pair<int32_t, float>> &b) {
    if (stage != 1 && stage != 2) {
      throw std::logic_error(
          "build_packed_subtree() needs load_vocabulary() first");
    }

    int32_t num_nodes = 0;
    id_node.clear();

    for (auto x : a) {
      auto id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }

    for (auto x : b) {
      auto id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }

    if (static_cast<int32_t>(subtree.size()) < num_nodes) {
      subtree.resize(num_nodes);
    }
    for (int32_t i = 0; i < num_nodes; ++i) {
      subtree[i].clear();
    }

    for (int32_t i = 0; i < num_nodes; ++i) {
      int32_t u = parents[id_node[i]];
      if (u != -1) {
        subtree[node_id[u]].push_back(i);
      }
    }

    if (static_cast<int32_t>(excess.size()) < num_nodes) {
      excess.resize(num_nodes);
    }

    delta_node.assign(num_nodes, 0.0f);
    unleaf.assign(num_nodes, -1);

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

    // [MOD] この時点で packed subtree の local node 数は id_node.size() で表される。
    // [MOD] 以後の getter はこのサイズを基準に有効範囲を返す。
  }

  // [MOD] Python から query-doc の 1対1 FlowTree 距離を直接呼べるようにする。
  // [MOD] load_dataset() を必須にせず、load_vocabulary() 済みなら使える。
  float flowtree_distance_pair(
      const std::vector<std::pair<int32_t, float>> &a,
      const std::vector<std::pair<int32_t, float>> &b) {
    if (stage != 1 && stage != 2) {
      throw std::logic_error("need to call load_vocabulary() first");
    }

    check_measure(a);
    check_measure(b);
    return flowtree_query(a, b);
  }

  // [MOD] 直前に build_packed_subtree(...) で構築した packed subtree の
  // [MOD] root(local node id) を Python 側から参照するための getter。
  int32_t get_last_packed_root() {
    if (id_node.empty()) {
      throw std::logic_error(
          "packed subtree is empty; call build_packed_subtree() first");
    }
    return node_id[0];
  }

  // [MOD] packed subtree の edge list を (parent, child) の配列として返す。
  // [MOD] Python 側で children 配列を再構成しやすくするための getter。
  std::vector<std::pair<int32_t, int32_t>> get_last_packed_edges() {
    std::vector<std::pair<int32_t, int32_t>> edges;
    for (int32_t u = 0; u < static_cast<int32_t>(id_node.size()); ++u) {
      for (auto v : subtree[u]) {
        edges.push_back(std::make_pair(u, v));
      }
    }
    return edges;
  }

  // [MOD] packed subtree の各 local node が葉かどうかを 0/1 配列で返す。
  std::vector<int32_t> get_last_packed_is_leaf() {
    std::vector<int32_t> is_leaf(id_node.size(), 0);
    for (int32_t u = 0; u < static_cast<int32_t>(id_node.size()); ++u) {
      is_leaf[u] = subtree[u].empty() ? 1 : 0;
    }
    return is_leaf;
  }

  // [MOD] packed subtree の各 local node に対応する語彙 id を返す。
  // [MOD] 葉でないノードは -1 のままになる想定。
  std::vector<int32_t> get_last_packed_unleaf() {
    std::vector<int32_t> out(id_node.size(), -1);
    for (int32_t u = 0; u < static_cast<int32_t>(id_node.size()); ++u) {
      out[u] = unleaf[u];
    }
    return out;
  }

  // [MOD] packed subtree の各 local node の差分質量 delta を返す。
  std::vector<float> get_last_packed_delta() {
    std::vector<float> out(id_node.size(), 0.0f);
    for (int32_t u = 0; u < static_cast<int32_t>(id_node.size()); ++u) {
      out[u] = delta_node[u];
    }
    return out;
  }

 private:
  std::vector<int32_t> parents;
  std::vector<int32_t> leaf;
  std::vector<int32_t> marked;
  int32_t num_queries;
  std::vector<int32_t> node_id;
  std::vector<int32_t> id_node;
  std::vector<std::vector<int32_t>> subtree;
  std::vector<std::vector<std::pair<float, int32_t>>> excess;
  std::vector<float> delta_node;
  std::unique_ptr<Matrix> dictionary;
  std::vector<int32_t> unleaf;
  std::vector<std::vector<std::pair<int32_t, float>>> dataset_embedding;
  std::vector<std::vector<std::pair<int32_t, float>>> raw_dataset;
  EigenMatrix means;
  EigenVector query_mean;
  std::vector<std::pair<float, int32_t>> distances;
  int32_t stage;

  void build_quadtree(const std::vector<int32_t> &subset,
                      const std::vector<std::pair<float, float>> &bounding_box,
                      int32_t depth, int32_t parent) {
    int32_t local_node_id = static_cast<int32_t>(parents.size());
    parents.push_back(parent);
    if (subset.size() == 1) {
      leaf[subset[0]] = local_node_id;
      return;
    }
    int32_t d = dictionary->cols();
    std::vector<float> mid(d);
    for (int32_t i = 0; i < d; ++i) {
      mid[i] = (bounding_box[i].first + bounding_box[i].second) / 2.0f;
    }
    std::map<std::vector<uint8_t>, std::vector<int32_t>> parts;
    for (auto ind : subset) {
      std::vector<uint8_t> code((d + 7) / 8, 0);
      for (int32_t i = 0; i < d; ++i) {
        if ((*dictionary)(ind, i) > mid[i]) {
          code[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
      }
      parts[code].push_back(ind);
    }
    std::vector<std::pair<float, float>> new_bounding_box(d);
    for (const auto &part : parts) {
      for (int32_t i = 0; i < d; ++i) {
        uint8_t bit = (part.first[i / 8] >> (i % 8)) & 1u;
        if (bit) {
          new_bounding_box[i] = std::make_pair(mid[i], bounding_box[i].second);
        } else {
          new_bounding_box[i] = std::make_pair(bounding_box[i].first, mid[i]);
        }
      }
      build_quadtree(part.second, new_bounding_box, depth + 1, local_node_id);
    }
  }

  std::vector<std::pair<int32_t, float>> compute_embedding(
      const std::vector<std::pair<int32_t, float>> &a) {
    std::vector<std::pair<int32_t, float>> result;
    for (auto x : a) {
      auto id = leaf[x.first];
      int32_t level = 0;
      while (id != -1) {
        ++level;
        id = parents[id];
      }
      id = leaf[x.first];
      while (id != -1) {
        --level;
        result.push_back(std::make_pair(id, x.second / (1 << level)));
        id = parents[id];
      }
    }
    std::sort(result.begin(), result.end());
    std::vector<std::pair<int32_t, float>> ans;
    for (auto x : result) {
      if (ans.empty() || ans.back().first != x.first) {
        ans.push_back(x);
      } else {
        ans.back().second += x.second;
      }
    }
    return ans;
  }

  float flowtree_query(const std::vector<std::pair<int32_t, float>> &a,
                       const std::vector<std::pair<int32_t, float>> &b) {
    // [MOD] packed subtree の構築を独立関数へ切り出した。
    build_packed_subtree(a, b);

    float res = run_query(0, node_id[0]);
    if (!excess[node_id[0]].empty()) {
      float unassigned = 0.0f;
      for (auto x : excess[node_id[0]]) {
        unassigned += x.first;
      }
      if (unassigned > EPS2) {
        throw std::logic_error("too much unassigned flow");
      }
    }
    ++num_queries;
    return res;
  }

  float run_query(int32_t depth, int32_t nd) {
    float res = 0.0f;
    for (auto x : subtree[nd]) {
      res += run_query(depth + 1, x);
    }
    excess[nd].clear();
    if (subtree[nd].empty()) {
      if (fabs(delta_node[nd]) > EPS) {
        excess[nd].push_back(std::make_pair(delta_node[nd], unleaf[nd]));
      }
    } else {
      for (auto x : subtree[nd]) {
        if (excess[x].empty()) {
          continue;
        }
        bool same = false;
        if (excess[nd].empty()) {
          same = true;
        } else if (sign(excess[x][0].first) == sign(excess[nd][0].first)) {
          same = true;
        }
        if (same) {
          for (auto y : excess[x]) {
            excess[nd].push_back(y);
          }
        } else {
          while (!excess[x].empty() && !excess[nd].empty()) {
            auto u = excess[nd].back();
            auto v = excess[x].back();

            float dist =
                ((*dictionary).row(u.second) - (*dictionary).row(v.second)).norm();
            if (fabs(u.first + v.first) < EPS) {
              excess[nd].pop_back();
              excess[x].pop_back();
              res += dist * fabs(u.first);
            } else if (fabs(u.first) < fabs(v.first)) {
              excess[nd].pop_back();
              excess[x].back().first += u.first;
              res += dist * fabs(u.first);
            } else {
              excess[x].pop_back();
              excess[nd].back().first += v.first;
              res += dist * fabs(v.first);
            }
          }
          if (!excess[x].empty()) {
            excess[x].swap(excess[nd]);
          }
        }
      }
    }
    (void)depth;
    return res;
  }

  void select_topk_aux(int32_t k1, NumPyIntArray output_ids,
                       NumPyFloatArray output_scores, bool to_sort) {
    auto output_ids_buf = output_ids.request();
    auto output_ids_raw = static_cast<int32_t *>(output_ids_buf.ptr);
    auto output_scores_buf = output_scores.request();
    auto output_scores_raw = static_cast<float *>(output_scores_buf.ptr);
    int32_t k2 = static_cast<int32_t>(output_ids_buf.shape[0]);
    std::nth_element(distances.begin(), distances.begin() + k2 - 1,
                     distances.begin() + k1);
    if (to_sort) {
      std::sort(distances.begin(), distances.begin() + k2);
    }
    for (int32_t i = 0; i < k2; ++i) {
      output_scores_raw[i] = distances[i].first;
      output_ids_raw[i] = distances[i].second;
    }
  }

  void check_measure(const std::vector<std::pair<int32_t, float>> &measure) {
    if (!dictionary) {
      throw std::logic_error("dictionary is not loaded");
    }
    float sum = 0.0f;
    auto n = dictionary->rows();
    for (auto &atom : measure) {
      if (atom.first < 0 || atom.first >= n) {
        throw std::logic_error("invalid index in the measure");
      }
      if (atom.second < -EPS) {
        throw std::logic_error("negative mass");
      }
      sum += atom.second;
    }
    //[MOD] 質量の総和が1でないことを許容する
    //if (fabs(sum - 1.0f) > EPS3) {
    //  throw std::logic_error("the masses don't sum to 1");
    //}
  }

  void check_stage() {
    if (stage != 2) {
      throw std::logic_error(
          "need to call load_vocabulary() and load_dataset() first");
    }
  }

  template <typename T>
  void check_dimension(T x) {
    auto buf = x.request();
    if (buf.ndim != 1) {
      throw std::logic_error(
          "input_ids, output_ids, output_scores must be one-dimensional");
    }
  }

  template <typename T>
  ssize_t get_length(T x) {
    return x.request().shape[0];
  }

  void check_input_output_arrays(NumPyIntArray input_ids,
                                 NumPyIntArray output_ids,
                                 NumPyFloatArray output_scores) {
    check_dimension(input_ids);
    check_dimension(output_ids);
    check_dimension(output_scores);
    auto l1 = get_length(input_ids);
    auto l2 = get_length(output_ids);
    auto l3 = get_length(output_scores);
    if (l2 != l3) {
      throw std::logic_error(
          "output_ids and output_scores must be of the same length");
    }
    if (l2 > l1) {
      throw std::logic_error(
          "output_ids and output_scores must be no longer than input_ids");
    }
    auto buf = static_cast<int32_t *>(input_ids.request().ptr);
    for (ssize_t i = 0; i < l1; ++i) {
      auto val = buf[i];
      if (val < 0 || val >= static_cast<int32_t>(raw_dataset.size())) {
        throw std::logic_error("input_ids contain an invalid index");
      }
    }
  }

  void check_input_output_arrays(NumPyIntArray input_ids,
                                 NumPyFloatArray input_scores,
                                 NumPyIntArray output_ids,
                                 NumPyFloatArray output_scores) {
    check_input_output_arrays(input_ids, output_ids, output_scores);
    check_dimension(input_scores);
    if (get_length(input_ids) != get_length(input_scores)) {
      throw std::logic_error(
          "input_ids and input_scores must be of the same length");
    }
  }
};

}  // namespace ote

PYBIND11_MODULE(ot_estimators_twotree, m) {
  using ote::OTEstimators;
  py::class_<OTEstimators>(m, "OTEstimators")
      .def(py::init<>())
      // [MOD] Python 側から load_vocabulary(vocab, seed) を呼べるようにする。
      .def("load_vocabulary", &OTEstimators::load_vocabulary,
        py::arg("points"), py::arg("seed") = -1)
      .def("load_dataset", &OTEstimators::load_dataset)
      .def("means_rank", &OTEstimators::means_rank)
      .def("overlap_rank", &OTEstimators::overlap_rank)
      .def("quadtree_rank", &OTEstimators::quadtree_rank)
      .def("flowtree_rank", &OTEstimators::flowtree_rank)
      .def("flowtree_distance_pair", &OTEstimators::flowtree_distance_pair)
      .def("build_packed_subtree", &OTEstimators::build_packed_subtree)
      .def("get_last_packed_root", &OTEstimators::get_last_packed_root)
      .def("get_last_packed_edges", &OTEstimators::get_last_packed_edges)
      .def("get_last_packed_is_leaf", &OTEstimators::get_last_packed_is_leaf)
      .def("get_last_packed_unleaf", &OTEstimators::get_last_packed_unleaf)
      .def("get_last_packed_delta", &OTEstimators::get_last_packed_delta)
      .def("select_topk", &OTEstimators::select_topk);
}