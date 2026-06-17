// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_vector_search` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_vector_search_init_seeds_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_init_seeds_f32(const float* queries, const float* dataset, const unsigned int* seed_indices, const unsigned int* n_queries, const unsigned int* n_seeds, const unsigned int* dim, const unsigned int* itopk_size, unsigned int* candidates, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int ns = n_seeds[0];
  unsigned int d = dim[0];
  unsigned int itopk = itopk_size[0];
  unsigned int total = (nq * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / ns);
    unsigned int seed_local = (gid % ns);
    unsigned int node_idx = seed_indices[seed_local];
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      unsigned int d_idx = ((node_idx * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    unsigned int out_idx = ((query_idx * itopk) + seed_local);
    candidates[out_idx] = node_idx;
    float final_dist = dist_acc;
    distances[out_idx] = final_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_expand_neighbors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_expand_neighbors_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* top_candidates, const unsigned int* n_queries, const unsigned int* search_width, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* new_candidates, float* new_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int sw = search_width[0];
  unsigned int gd = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int neighbors_per_query = (sw * gd);
  unsigned int total = (nq * neighbors_per_query);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / neighbors_per_query);
    unsigned int local_idx = (gid % neighbors_per_query);
    unsigned int candidate_idx = (local_idx / gd);
    unsigned int neighbor_idx = (local_idx % gd);
    unsigned int tc_idx = ((query_idx * sw) + candidate_idx);
    unsigned int parent_node = top_candidates[tc_idx];
    unsigned int graph_idx = ((parent_node * gd) + neighbor_idx);
    unsigned int neighbor_node = graph[graph_idx];
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      unsigned int d_idx = ((neighbor_node * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    unsigned int out_idx = ((query_idx * neighbors_per_query) + local_idx);
    new_candidates[out_idx] = neighbor_node;
    float final_dist = dist_acc;
    new_distances[out_idx] = final_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_check_visited = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_check_visited(const unsigned int* candidates, unsigned int* visited, const unsigned int* n, const unsigned int* hash_size, unsigned int* is_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int num = n[0];
  unsigned int hs = hash_size[0];
  bool inBounds = (gid < num);
  if (inBounds) {
    unsigned int node = candidates[gid];
    unsigned int hash_idx = (node % hs);
    unsigned int old_val = atomicCAS((int*)(&(visited[hash_idx])), (int)(0u), (int)((node + 1u)));
    bool was_empty = (old_val == 0u);
    unsigned int result = ((was_empty) ? (1u) : (0u));
    is_new[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_clear_visited = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_clear_visited(const unsigned int* hash_size, unsigned int* visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int hs = hash_size[0];
  bool inBounds = (gid < hs);
  if (inBounds) {
    visited[gid] = 0u;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_extract_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_extract_results_f32(const unsigned int* sorted_candidates, const float* sorted_distances, const unsigned int* n_queries, const unsigned int* k, const unsigned int* itopk_size, unsigned int* result_indices, float* result_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int kVal = k[0];
  unsigned int itopk = itopk_size[0];
  unsigned int total = (nq * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int query_idx = (gid / kVal);
    unsigned int k_idx = (gid % kVal);
    unsigned int src_idx = ((query_idx * itopk) + k_idx);
    unsigned int idx_val = sorted_candidates[src_idx];
    float dist_val = sorted_distances[src_idx];
    result_indices[gid] = idx_val;
    result_distances[gid] = dist_val;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_distances_batch_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_distances_batch_f32(const float* query, const float* candidates, const unsigned int* candidate_ids, const unsigned int* n_candidates, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_cand = n_candidates[0];
  unsigned int d = dim[0];
  if ((gid < n_cand)) {
    unsigned int cand_id = candidate_ids[gid];
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      unsigned int c_idx = ((cand_id * d) + k);
      float c_val = candidates[c_idx];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_acc = acc;
    distances[gid] = final_acc;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_search_layer_greedy_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_search_layer_greedy_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* graph_degrees, const unsigned int* entry_points, const unsigned int* n_queries, const unsigned int* dim, const unsigned int* ef, const unsigned int* max_degree, unsigned int* candidates, float* candidate_distances, unsigned int* n_found) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int d = dim[0];
  if ((query_id < nq)) {
    unsigned int entry = entry_points[query_id];
    float best_dist = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_id * d) + k);
      unsigned int e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_bd = best_dist;
      best_dist = (old_bd + sq);
    }
    unsigned int ef_val = ef[0];
    unsigned int cand_idx = (query_id * ef_val);
    candidates[cand_idx] = entry;
    float init_dist = best_dist;
    candidate_distances[cand_idx] = init_dist;
    n_found[query_id] = 1u;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_select_neighbors_simple_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_select_neighbors_simple_f32(const unsigned int* candidates, const float* distances, const unsigned int* n_candidates, const unsigned int* m, unsigned int* selected, unsigned int* selected_count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_cand = n_candidates[0];
  unsigned int m_val = m[0];
  if ((tid == 0u)) {
    unsigned int limit = (((n_cand < m_val)) ? (n_cand) : (m_val));
    for (int i = 0; i < (int)(limit); ++i) {
      unsigned int cand = candidates[i];
      selected[i] = cand;
    }
    selected_count[0u] = limit;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_select_neighbors_heuristic_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_select_neighbors_heuristic_f32(const float* query, const unsigned int* candidates, const float* distances, const float* dataset, const unsigned int* n_candidates, const unsigned int* dim, const unsigned int* m, unsigned int* selected, unsigned int* n_selected) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_cand = n_candidates[0];
  unsigned int m_val = m[0];
  __shared__ unsigned int selected_ids[64];
  __shared__ unsigned int n_sel[1];
  if ((tid == 0u)) {
    n_sel[0u] = 0u;
  }
  __syncthreads();
  if ((tid < n_cand)) {
    unsigned int cand_id = candidates[tid];
    selected[tid] = cand_id;
  }
  __syncthreads();
  if ((tid == 0u)) {
    unsigned int final_count = (((n_cand < m_val)) ? (n_cand) : (m_val));
    n_selected[0u] = final_count;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_update_candidate_list_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_update_candidate_list_f32(unsigned int* candidates, float* distances, const unsigned int* new_candidates, const float* new_distances, const unsigned int* n_current, const unsigned int* n_new, const unsigned int* ef, unsigned int* n_merged) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n_curr = n_current[0];
  unsigned int n_new_val = n_new[0];
  unsigned int ef_val = ef[0];
  __shared__ unsigned int merge_ids[512];
  __shared__ float merge_dists[512];
  if ((tid < n_curr)) {
    unsigned int cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  __syncthreads();
  if ((tid < n_new_val)) {
    unsigned int offset = (n_curr + tid);
    unsigned int nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  __syncthreads();
  if ((tid == 0u)) {
    unsigned int total = (n_curr + n_new_val);
    unsigned int final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[0u] = final_n;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_extract_layer_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_extract_layer_results_f32(const unsigned int* candidates, const float* distances, const unsigned int* n_found, const unsigned int* n_queries, const unsigned int* k, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int query_id = (gid / k_val);
  unsigned int neighbor_idx = (gid % k_val);
  if ((query_id < nq)) {
    unsigned int found = n_found[query_id];
    unsigned int src_idx = ((query_id * k_val) + neighbor_idx);
    if ((neighbor_idx < found)) {
      unsigned int cand_id = candidates[src_idx];
      float cand_dist = distances[src_idx];
      result_ids[src_idx] = cand_id;
      result_dists[src_idx] = cand_dist;
    } else {
      result_ids[src_idx] = 4294967295u;
      result_dists[src_idx] = __uint_as_float(0x7e967699u);
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_init_search_state_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_init_search_state_f32(const unsigned int* global_entry, const unsigned int* n_queries, const unsigned int* visited_size, unsigned int* entry_points, unsigned int* visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int entry = global_entry[0];
  unsigned int nq = n_queries[0];
  unsigned int vs = visited_size[0];
  if ((gid < nq)) {
    entry_points[gid] = entry;
  }
  unsigned int total_visited = (nq * vs);
  if ((gid < total_visited)) {
    unsigned int q_idx = (gid / vs);
    unsigned int v_idx = (gid % vs);
    unsigned int flat_idx = ((q_idx * vs) + v_idx);
    visited[flat_idx] = 0u;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_mark_visited_batch = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_mark_visited_batch(const unsigned int* node_ids, const unsigned int* query_ids, const unsigned int* n_nodes, const unsigned int* visited_stride, unsigned int* visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int stride = visited_stride[0];
  if ((tid < nn)) {
    unsigned int node = node_ids[tid];
    unsigned int query = query_ids[tid];
    unsigned int word_idx = (node / 32u);
    unsigned int bit_idx = (node % 32u);
    unsigned int bit_mask = (1u << bit_idx);
    unsigned int flat_idx = ((query * stride) + word_idx);
    atomicOr(&visited[flat_idx], bit_mask);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_check_visited_batch = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_check_visited_batch(const unsigned int* node_ids, const unsigned int* query_ids, const unsigned int* n_nodes, const unsigned int* visited_stride, const unsigned int* visited, unsigned int* is_visited) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int stride = visited_stride[0];
  if ((tid < nn)) {
    unsigned int node = node_ids[tid];
    unsigned int query = query_ids[tid];
    unsigned int word_idx = (node / 32u);
    unsigned int bit_idx = (node % 32u);
    unsigned int bit_mask = (1u << bit_idx);
    unsigned int flat_idx = ((query * stride) + word_idx);
    unsigned int word = visited[flat_idx];
    unsigned int is_set = (word & bit_mask);
    unsigned int result = (((is_set != 0u)) ? (1u) : (0u));
    is_visited[tid] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_merge_multi_query_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_merge_multi_query_results_f32(const unsigned int* layer_results, const float* layer_distances, const unsigned int* n_queries, const unsigned int* k, const unsigned int* n_layers, unsigned int* final_ids, float* final_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int query_id = (gid / k_val);
  unsigned int k_idx = (gid % k_val);
  if ((query_id < nq)) {
    unsigned int flat_idx = ((query_id * k_val) + k_idx);
    unsigned int result_id = layer_results[flat_idx];
    float result_dist = layer_distances[flat_idx];
    final_ids[flat_idx] = result_id;
    final_distances[flat_idx] = result_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_neighbor_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_neighbor_distances_f32(const float* query, const float* dataset, const unsigned int* neighbor_ids, const unsigned int* n_neighbors, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_neighbors[0];
  unsigned int d = dim[0];
  if ((tid < nn)) {
    unsigned int neighbor_id = neighbor_ids[tid];
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      unsigned int n_idx = ((neighbor_id * d) + k);
      float n_val = dataset[n_idx];
      float diff = (q_val - n_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_dist = acc;
    distances[tid] = final_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_greedy_search_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_greedy_search_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* graph_degrees, const unsigned int* entry_point, const unsigned int* n_queries, const unsigned int* dim, const unsigned int* max_degree, const unsigned int* search_list_size, unsigned int* candidates, float* candidate_dists, unsigned int* n_candidates) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int d = dim[0];
  unsigned int entry = entry_point[0];
  unsigned int ls = search_list_size[0];
  if ((query_id < nq)) {
    float entry_dist = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((query_id * d) + k);
      unsigned int e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_ed = entry_dist;
      entry_dist = (old_ed + sq);
    }
    unsigned int cand_base = (query_id * ls);
    candidates[cand_base] = entry;
    float ed = entry_dist;
    candidate_dists[cand_base] = ed;
    n_candidates[query_id] = 1u;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_robust_prune_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_robust_prune_f32(const unsigned int* vertex, const unsigned int* candidates, const float* distances, const float* dataset, const unsigned int* n_candidates, const unsigned int* dim, const unsigned int* max_degree, const unsigned int* alpha, unsigned int* selected, unsigned int* n_selected) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nc = n_candidates[0];
  unsigned int md = max_degree[0];
  if ((tid == 0u)) {
    unsigned int limit = (((nc < md)) ? (nc) : (md));
    for (int i = 0; i < (int)(limit); ++i) {
      unsigned int cand = candidates[i];
      selected[i] = cand;
    }
    n_selected[0u] = limit;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_insert_vertex_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_insert_vertex_f32(const unsigned int* vertex_id, const unsigned int* candidates, const unsigned int* n_candidates, const unsigned int* max_degree, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int vid = vertex_id[0];
  unsigned int nc = n_candidates[0];
  unsigned int md = max_degree[0];
  unsigned int n_neighbors = (((nc < md)) ? (nc) : (md));
  if ((tid < n_neighbors)) {
    unsigned int neighbor = candidates[tid];
    unsigned int graph_idx = ((vid * md) + tid);
    graph[graph_idx] = neighbor;
  }
  if ((tid == 0u)) {
    graph_degrees[vid] = n_neighbors;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_update_neighbors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_update_neighbors_f32(const unsigned int* new_vertex, const unsigned int* new_neighbors, const unsigned int* n_new_neighbors, const unsigned int* max_degree, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = new_vertex[0];
  unsigned int nn = n_new_neighbors[0];
  unsigned int md = max_degree[0];
  if ((tid < nn)) {
    unsigned int neighbor = new_neighbors[tid];
    unsigned int old_degree = atomicAdd(&(graph_degrees[neighbor]), 1u);
    if ((old_degree < md)) {
      unsigned int graph_idx = ((neighbor * md) + old_degree);
      graph[graph_idx] = nv;
    } else {
      unsigned int _ignore = atomicAdd(&(graph_degrees[neighbor]), -(1u));
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_random_init_neighbors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_random_init_neighbors_f32(const unsigned int* n_vectors, const unsigned int* max_degree, const unsigned int* init_degree, const unsigned int* seed, unsigned int* graph, unsigned int* graph_degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int md = max_degree[0];
  unsigned int id = init_degree[0];
  unsigned int s = seed[0];
  if ((vid < nv)) {
    unsigned int rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      unsigned int old_rng = rng;
      unsigned int temp1 = (old_rng * 1103515245u);
      unsigned int temp2 = (temp1 + 12345u);
      unsigned int new_rng = (temp2 & 2147483647u);
      rng = new_rng;
      unsigned int neighbor_raw = (new_rng % nv);
      unsigned int neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + 1u) % nv)) : (neighbor_raw));
      unsigned int graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_extract_search_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_extract_search_results_f32(const unsigned int* candidates, const float* candidate_dists, const unsigned int* n_candidates, const unsigned int* n_queries, const unsigned int* k, const unsigned int* search_list_size, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int ls = search_list_size[0];
  unsigned int query_id = (gid / k_val);
  unsigned int k_idx = (gid % k_val);
  if ((query_id < nq)) {
    unsigned int nc = n_candidates[query_id];
    unsigned int src_idx = ((query_id * ls) + k_idx);
    unsigned int dst_idx = ((query_id * k_val) + k_idx);
    if ((k_idx < nc)) {
      unsigned int cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = 4294967295u;
      result_dists[dst_idx] = __uint_as_float(0x7e967699u);
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_batched_greedy_search_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_batched_greedy_search_f32(const float* queries, const float* dataset, const unsigned int* graph, const unsigned int* graph_degrees, const unsigned int* entry_point, const unsigned int* n_queries, const unsigned int* n_vectors, const unsigned int* dim, const unsigned int* max_degree, const unsigned int* k, unsigned int* result_ids, float* result_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x)));
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int d = dim[0];
  unsigned int entry = entry_point[0];
  unsigned int k_val = k[0];
  __shared__ float partial_sums[64];
  if ((query_id < nq)) {
    if ((tid == 0u)) {
      unsigned int base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = __uint_as_float(0x00000000u);
    }
    if ((tid < k_val)) {
      if ((tid != 0u)) {
        unsigned int out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = 4294967295u;
        result_dists[out_idx] = __uint_as_float(0x7e967699u);
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_centroid_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_centroid_distances_f32(const float* queries, const float* centroids, const unsigned int* n_queries, const unsigned int* n_lists, const unsigned int* dim, float* centroid_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int nl = n_lists[0];
  unsigned int d = dim[0];
  unsigned int total_pairs = (nq * nl);
  bool inBounds = (gid < total_pairs);
  if (inBounds) {
    unsigned int query_idx = (gid / nl);
    unsigned int list_idx = (gid % nl);
    unsigned int query_base = (query_idx * d);
    unsigned int centroid_base = (list_idx * d);
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_offset = (query_base + k);
      unsigned int c_offset = (centroid_base + k);
      float q_val = queries[q_offset];
      float c_val = centroids[c_offset];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_acc = acc;
    centroid_distances[gid] = final_acc;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_select_probes_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_select_probes_f32(const float* centroid_distances, const unsigned int* n_queries, const unsigned int* n_lists, const unsigned int* n_probes, unsigned int* probe_indices, float* probe_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int query_idx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int nl = n_lists[0];
  unsigned int np = n_probes[0];
  bool query_valid = (query_idx < nq);
  if (query_valid) {
    extern __shared__ float s_distances[]; /* dynamic shared memory (best-effort) */
    extern __shared__ unsigned int s_indices[]; /* dynamic shared memory (best-effort) */
    s_distances[tid] = __uint_as_float(0x7f7fc99eu);
    s_indices[tid] = 0u;
    unsigned int base = (query_idx * nl);
    float local_min_dist = __uint_as_float(0x7f7fc99eu);
    unsigned int local_min_idx = 0u;
    unsigned int iters = ((nl / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int list_idx = ((iter * wg_size) + tid);
      bool valid = (list_idx < nl);
      if (valid) {
        unsigned int offset = (base + list_idx);
        float dist = centroid_distances[offset];
        float cur_min = local_min_dist;
        bool is_better = (dist < cur_min);
        if (is_better) {
          local_min_dist = dist;
          local_min_idx = list_idx;
        }
      }
    }
    float final_min_dist = local_min_dist;
    unsigned int final_min_idx = local_min_idx;
    s_distances[tid] = final_min_dist;
    s_indices[tid] = final_min_idx;
    __syncthreads();
    bool is_thread0 = (tid == 0u);
    if (is_thread0) {
      for (int p = 0; p < (int)(np); ++p) {
        float best_dist = __uint_as_float(0x7f7fc99eu);
        unsigned int best_idx = 0u;
        unsigned int best_slot = 0u;
        for (int s = 0; s < (int)(wg_size); ++s) {
          float curr_dist = s_distances[s];
          unsigned int curr_idx = s_indices[s];
          float cur_best = best_dist;
          bool better = (curr_dist < cur_best);
          if (better) {
            best_dist = curr_dist;
            best_idx = curr_idx;
            best_slot = s;
          }
        }
        unsigned int out_base = (query_idx * np);
        unsigned int out_offset = (out_base + p);
        unsigned int sel_idx = best_idx;
        float sel_dist = best_dist;
        probe_indices[out_offset] = sel_idx;
        probe_distances[out_offset] = sel_dist;
        unsigned int slot_to_mark = best_slot;
        s_distances[slot_to_mark] = __uint_as_float(0x7f7fc99eu);
      }
    }
    __syncthreads();
  }
}

)kernel";
const char* k_hip_navatala_vector_search_scan_inverted_list_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_scan_inverted_list_f32(const float* queries, const float* list_data, const unsigned int* list_indices, const unsigned int* list_offsets, const unsigned int* probe_indices, const unsigned int* n_queries, const unsigned int* n_probes, const unsigned int* dim, const unsigned int* max_list_size, float* scan_distances, unsigned int* scan_indices, unsigned int* scan_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int np = n_probes[0];
  unsigned int d = dim[0];
  unsigned int mls = max_list_size[0];
  unsigned int total_pairs = (nq * np);
  bool valid_wg = (wgid < total_pairs);
  if (valid_wg) {
    unsigned int query_idx = (wgid / np);
    unsigned int probe_idx = (wgid % np);
    unsigned int probe_offset = ((query_idx * np) + probe_idx);
    unsigned int list_idx = probe_indices[probe_offset];
    unsigned int list_start = list_offsets[list_idx];
    unsigned int list_idx_plus1 = (list_idx + 1u);
    unsigned int list_end = list_offsets[list_idx_plus1];
    unsigned int list_size = (list_end - list_start);
    unsigned int query_base = (query_idx * d);
    unsigned int out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    unsigned int iters = ((list_size / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        unsigned int global_vec_idx = (list_start + local_vec_idx);
        unsigned int vec_base = (global_vec_idx * d);
        float dist_acc = __uint_as_float(0x00000000u);
        for (int k = 0; k < (int)(d); ++k) {
          unsigned int q_off = (query_base + k);
          unsigned int v_off = (vec_base + k);
          float q_val = queries[q_off];
          float v_val = list_data[v_off];
          float diff = (q_val - v_val);
          float sq = (diff * diff);
          float old_d = dist_acc;
          dist_acc = (old_d + sq);
        }
        unsigned int orig_idx = list_indices[global_vec_idx];
        unsigned int out_offset = (out_base + local_vec_idx);
        float final_d = dist_acc;
        scan_distances[out_offset] = final_d;
        scan_indices[out_offset] = orig_idx;
      }
    }
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      unsigned int count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_merge_probe_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_merge_probe_results_f32(const float* scan_distances, const unsigned int* scan_indices, const unsigned int* scan_counts, const unsigned int* n_queries, const unsigned int* n_probes, const unsigned int* max_list_size, const unsigned int* k, float* final_distances, unsigned int* final_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int query_idx = ((unsigned int)((int)(blockIdx.x)));
  unsigned int wg_size = 256u;
  unsigned int nq = n_queries[0];
  unsigned int np = n_probes[0];
  unsigned int mls = max_list_size[0];
  unsigned int kval = k[0];
  bool valid_query = (query_idx < nq);
  if (valid_query) {
    extern __shared__ float s_best_dists[]; /* dynamic shared memory (best-effort) */
    extern __shared__ unsigned int s_best_idxs[]; /* dynamic shared memory (best-effort) */
    s_best_dists[tid] = __uint_as_float(0x7f7fc99eu);
    s_best_idxs[tid] = 0u;
    unsigned int total_max = (np * mls);
    unsigned int iters = ((total_max / wg_size) + 1u);
    float my_best_dist = __uint_as_float(0x7f7fc99eu);
    unsigned int my_best_idx = 0u;
    for (int iter = 0; iter < (int)(iters); ++iter) {
      unsigned int flat_idx = ((iter * wg_size) + tid);
      bool valid_idx = (flat_idx < total_max);
      if (valid_idx) {
        unsigned int probe_idx = (flat_idx / mls);
        unsigned int local_vec = (flat_idx % mls);
        unsigned int count_offset = ((query_idx * np) + probe_idx);
        unsigned int actual_count = scan_counts[count_offset];
        bool within_count = (local_vec < actual_count);
        if (within_count) {
          unsigned int scan_offset = (((query_idx * np) * mls) + flat_idx);
          float dist = scan_distances[scan_offset];
          unsigned int idx = scan_indices[scan_offset];
          float cur_best = my_best_dist;
          bool better = (dist < cur_best);
          if (better) {
            my_best_dist = dist;
            my_best_idx = idx;
          }
        }
      }
    }
    float my_d = my_best_dist;
    unsigned int my_i = my_best_idx;
    s_best_dists[tid] = my_d;
    s_best_idxs[tid] = my_i;
    __syncthreads();
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      for (int ki = 0; ki < (int)(kval); ++ki) {
        float sel_dist = __uint_as_float(0x7f7fc99eu);
        unsigned int sel_idx = 0u;
        unsigned int sel_slot = 0u;
        for (int s = 0; s < (int)(wg_size); ++s) {
          float sd = s_best_dists[s];
          unsigned int si = s_best_idxs[s];
          float cur_sel = sel_dist;
          bool b = (sd < cur_sel);
          if (b) {
            sel_dist = sd;
            sel_idx = si;
            sel_slot = s;
          }
        }
        unsigned int out_off = ((query_idx * kval) + ki);
        unsigned int final_sel_idx = sel_idx;
        float final_sel_dist = sel_dist;
        final_distances[out_off] = final_sel_dist;
        final_indices[out_off] = final_sel_idx;
        unsigned int slot_mark = sel_slot;
        s_best_dists[slot_mark] = __uint_as_float(0x7f7fc99eu);
      }
    }
    __syncthreads();
  }
}

)kernel";
const char* k_hip_navatala_vector_search_assign_to_cluster_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_assign_to_cluster_f32(const float* vectors, const float* centroids, const unsigned int* n_vectors, const unsigned int* n_lists, const unsigned int* dim, unsigned int* labels) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int nl = n_lists[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    unsigned int vec_base = (gid * d);
    float best_dist = __uint_as_float(0x7f7fc99eu);
    unsigned int best_label = 0u;
    for (int c = 0; c < (int)(nl); ++c) {
      unsigned int cent_base = (c * d);
      float dist_acc = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        unsigned int v_off = (vec_base + k);
        unsigned int c_off = (cent_base + k);
        float v_val = vectors[v_off];
        float c_val = centroids[c_off];
        float diff = (v_val - c_val);
        float sq = (diff * diff);
        float old_da = dist_acc;
        dist_acc = (old_da + sq);
      }
      float final_da = dist_acc;
      float cur_best = best_dist;
      bool better = (final_da < cur_best);
      if (better) {
        best_dist = final_da;
        best_label = c;
      }
    }
    unsigned int final_label = best_label;
    labels[gid] = final_label;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_count_cluster_sizes = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_count_cluster_sizes(const unsigned int* labels, const unsigned int* n_vectors, const unsigned int* n_lists, unsigned int* cluster_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    unsigned int label = labels[gid];
    unsigned int _old_count = atomicAdd(&(cluster_counts[label]), 1u);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_populate_inverted_lists_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_populate_inverted_lists_f32(const float* vectors, const unsigned int* labels, const unsigned int* list_offsets, const unsigned int* n_vectors, const unsigned int* dim, float* list_data, unsigned int* list_indices, unsigned int* write_positions) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    unsigned int label = labels[gid];
    unsigned int list_base = list_offsets[label];
    unsigned int local_pos = atomicAdd(&(write_positions[label]), 1u);
    unsigned int global_pos = (list_base + local_pos);
    unsigned int src_base = (gid * d);
    unsigned int dst_base = (global_pos * d);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int src_off = (src_base + k);
      unsigned int dst_off = (dst_base + k);
      float val = vectors[src_off];
      list_data[dst_off] = val;
    }
    list_indices[global_pos] = gid;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_initialize_random_graph = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_initialize_random_graph(const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* seed, unsigned int* graph, float* graph_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int s = seed[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int slot = (gid % deg);
    unsigned int state0 = ((gid * 1103515245u) + (s + 12345u));
    unsigned int rand_raw = (state0 % nn);
    bool is_self = (rand_raw == node_idx);
    unsigned int alt = ((node_idx + 1u) % nn);
    unsigned int neighbor = ((is_self) ? (alt) : (rand_raw));
    graph[gid] = neighbor;
    graph_distances[gid] = __uint_as_float(0x7f7fc99eu);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_graph_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_graph_distances_f32(const float* dataset, const unsigned int* graph, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, float* graph_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int neighbor = graph[gid];
    unsigned int node_base = (node_idx * d);
    unsigned int neighbor_base = (neighbor * d);
    float dist_acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int n_off = (node_base + k);
      unsigned int nb_off = (neighbor_base + k);
      float n_val = dataset[n_off];
      float nb_val = dataset[nb_off];
      float diff = (n_val - nb_val);
      float sq = (diff * diff);
      float old_d = dist_acc;
      dist_acc = (old_d + sq);
    }
    float final_d = dist_acc;
    graph_distances[gid] = final_d;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_nn_descent_join_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_nn_descent_join_f32(const float* dataset, const unsigned int* graph, const float* graph_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* new_graph, float* new_distances, unsigned int* update_count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int node_idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    unsigned int node_base = (node_idx * d);
    unsigned int graph_base = (node_idx * deg);
    float worst_dist = __uint_as_float(0x00000000u);
    unsigned int worst_slot = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int slot_idx = (graph_base + i);
      float slot_dist = graph_distances[slot_idx];
      float cur_worst = worst_dist;
      bool is_worse = (slot_dist > cur_worst);
      if (is_worse) {
        worst_dist = slot_dist;
        worst_slot = i;
      }
    }
    for (int ni = 0; ni < (int)(deg); ++ni) {
      unsigned int neighbor_idx_slot = (graph_base + ni);
      unsigned int neighbor = graph[neighbor_idx_slot];
      unsigned int neighbor_graph_base = (neighbor * deg);
      for (int nni = 0; nni < (int)(deg); ++nni) {
        unsigned int nn_slot = (neighbor_graph_base + nni);
        unsigned int candidate = graph[nn_slot];
        bool is_self = (candidate == node_idx);
        if (((candidate != node_idx) && true)) {
          unsigned int cand_base = (candidate * d);
          float cand_dist_acc = __uint_as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            unsigned int n_off = (node_base + k);
            unsigned int c_off = (cand_base + k);
            float n_val = dataset[n_off];
            float c_val = dataset[c_off];
            float diff = (n_val - c_val);
            float sq = (diff * diff);
            float old_cd = cand_dist_acc;
            cand_dist_acc = (old_cd + sq);
          }
          float cand_dist = cand_dist_acc;
          float cur_worst2 = worst_dist;
          bool is_better = (cand_dist < cur_worst2);
          if (is_better) {
            unsigned int update_slot = worst_slot;
            unsigned int update_idx = (graph_base + update_slot);
            new_graph[update_idx] = candidate;
            new_distances[update_idx] = cand_dist;
            worst_dist = cand_dist;
            unsigned int _upd = atomicAdd(&(update_count[0u]), 1u);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_build_reverse_graph = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_build_reverse_graph(const unsigned int* graph, const unsigned int* n_nodes, const unsigned int* graph_degree, unsigned int* reverse_graph, unsigned int* reverse_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int source = (gid / deg);
    unsigned int target = graph[gid];
    unsigned int rev_slot = atomicAdd(&(reverse_counts[target]), 1u);
    bool has_space = (rev_slot < deg);
    if (has_space) {
      unsigned int rev_idx = ((target * deg) + rev_slot);
      reverse_graph[rev_idx] = source;
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_prune_graph_r_n_g_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_prune_graph_r_n_g_f32(const float* dataset, const unsigned int* graph, const float* graph_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, const unsigned int* dim, unsigned int* pruned_graph, float* pruned_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  unsigned int d = dim[0];
  unsigned int total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int node_idx = (gid / deg);
    unsigned int slot = (gid % deg);
    unsigned int graph_base = (node_idx * deg);
    unsigned int target = graph[gid];
    float edge_dist = graph_distances[gid];
    unsigned int target_base = (target * d);
    bool should_prune = false;
    for (int wi = 0; wi < (int)(deg); ++wi) {
      bool is_self_slot = (wi == slot);
      if ((wi != slot)) {
        unsigned int w_slot = (graph_base + wi);
        unsigned int w = graph[w_slot];
        float d_uw = graph_distances[w_slot];
        bool uw_closer = (d_uw < edge_dist);
        if (uw_closer) {
          unsigned int w_base = (w * d);
          float d_wv_acc = __uint_as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            unsigned int w_off = (w_base + k);
            unsigned int t_off = (target_base + k);
            float w_val = dataset[w_off];
            float t_val = dataset[t_off];
            float diff = (w_val - t_val);
            float sq = (diff * diff);
            float old_wv = d_wv_acc;
            d_wv_acc = (old_wv + sq);
          }
          float d_wv = d_wv_acc;
          bool wv_closer = (d_wv < edge_dist);
          if (wv_closer) {
            should_prune = true;
          }
        }
      }
    }
    bool prune = should_prune;
    unsigned int invalid_marker = 4294967295u;
    unsigned int out_idx = ((prune) ? (invalid_marker) : (target));
    float out_dist = ((prune) ? (__uint_as_float(0x7f7fc99eu)) : (edge_dist));
    pruned_graph[gid] = out_idx;
    pruned_distances[gid] = out_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compact_pruned_graph = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compact_pruned_graph(const unsigned int* pruned_graph, const float* pruned_distances, const unsigned int* n_nodes, const unsigned int* graph_degree, unsigned int* compact_graph, float* compact_distances, unsigned int* degree_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int node_idx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_nodes[0];
  unsigned int deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    unsigned int base = (node_idx * deg);
    unsigned int write_pos = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int read_idx = (base + i);
      unsigned int neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != 4294967295u);
      if (is_valid) {
        unsigned int wp = write_pos;
        unsigned int write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + 1u);
      }
    }
    unsigned int final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_vertex_degrees = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_vertex_degrees(const unsigned int* graph, const unsigned int* n_vertices, const unsigned int* max_degree, const unsigned int* invalid_id, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int md = max_degree[0];
  unsigned int inv = invalid_id[0];
  if ((vid < nv)) {
    unsigned int count = 0u;
    for (int i = 0; i < (int)(md); ++i) {
      unsigned int idx = ((vid * md) + i);
      unsigned int neighbor = graph[idx];
      if ((neighbor != inv)) {
        unsigned int old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    unsigned int final_count = count;
    degrees[vid] = final_count;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_prune_excess_degree = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_prune_excess_degree(unsigned int* graph, const float* neighbor_distances, unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* current_max_degree, const unsigned int* target_max_degree, const unsigned int* invalid_id) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int cmd = current_max_degree[0];
  unsigned int tmd = target_max_degree[0];
  unsigned int inv = invalid_id[0];
  if ((vid < nv)) {
    unsigned int deg = degrees[vid];
    if ((deg > tmd)) {
      for (int i = 0; i < (int)(cmd); ++i) {
        if ((i >= tmd)) {
          unsigned int idx = ((vid * cmd) + i);
          graph[idx] = inv;
        }
      }
      degrees[vid] = tmd;
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_reorder_vertices_by_degree = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_reorder_vertices_by_degree(const unsigned int* degrees, unsigned int* prefix_sums, const unsigned int* n_vertices, const unsigned int* n_bins, unsigned int* old_to_new, unsigned int* new_to_old) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int nb = n_bins[0];
  if ((vid < nv)) {
    unsigned int deg = degrees[vid];
    unsigned int bin = (((deg >= nb)) ? ((nb - 1u)) : (deg));
    unsigned int new_id = atomicAdd(&(prefix_sums[bin]), 1u);
    old_to_new[vid] = new_id;
    new_to_old[new_id] = vid;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compact_graph_edges = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compact_graph_edges(unsigned int* graph, const unsigned int* n_vertices, const unsigned int* max_degree, const unsigned int* invalid_id, unsigned int* degrees) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int md = max_degree[0];
  unsigned int inv = invalid_id[0];
  if ((vid < nv)) {
    unsigned int write_pos = 0u;
    for (int i = 0; i < (int)(md); ++i) {
      unsigned int read_idx = ((vid * md) + i);
      unsigned int neighbor = graph[read_idx];
      if ((neighbor != inv)) {
        unsigned int wp = write_pos;
        unsigned int write_idx = ((vid * md) + wp);
        graph[write_idx] = neighbor;
        write_pos = (wp + 1u);
      }
    }
    unsigned int final_wp = write_pos;
    for (int j = 0; j < (int)(md); ++j) {
      if ((j >= final_wp)) {
        unsigned int fill_idx = ((vid * md) + j);
        graph[fill_idx] = inv;
      }
    }
    degrees[vid] = final_wp;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_validate_graph_integrity = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_validate_graph_integrity(const unsigned int* graph, const unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* max_degree, unsigned int* error_flags) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int md = max_degree[0];
  if ((vid < nv)) {
    unsigned int errors = 0u;
    unsigned int deg = degrees[vid];
    for (int i = 0; i < (int)(deg); ++i) {
      unsigned int idx = ((vid * md) + i);
      unsigned int neighbor = graph[idx];
      if ((neighbor == vid)) {
        unsigned int old_err = errors;
        errors = (old_err | 1u);
      }
      if ((neighbor >= nv)) {
        unsigned int old_err2 = errors;
        errors = (old_err2 | 2u);
      }
    }
    unsigned int final_errors = errors;
    error_flags[vid] = final_errors;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_prune_excess_degree_sorted = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_prune_excess_degree_sorted(unsigned int* graph, const float* neighbor_distances, unsigned int* degrees, const unsigned int* n_vertices, const unsigned int* current_max_degree, const unsigned int* target_max_degree, const unsigned int* invalid_id) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x)));
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nv = n_vertices[0];
  unsigned int cmd = current_max_degree[0];
  unsigned int tmd = target_max_degree[0];
  unsigned int inv = invalid_id[0];
  __shared__ unsigned int shared_ids[128];
  __shared__ float shared_dists[128];
  if ((vid < nv)) {
    unsigned int deg = degrees[vid];
    if ((deg > tmd)) {
      if ((tid < deg)) {
        unsigned int read_idx = ((vid * cmd) + tid);
        unsigned int neighbor_id = graph[read_idx];
        float neighbor_dist = neighbor_distances[read_idx];
        shared_ids[tid] = neighbor_id;
        shared_dists[tid] = neighbor_dist;
      } else {
        if ((tid < 128u)) {
          shared_ids[tid] = inv;
          shared_dists[tid] = __uint_as_float(0x7e967699u);
        }
      }
      __syncthreads();
      for (int stage = 0; stage < (int)(7u); ++stage) {
        for (int substage = 0; substage < (int)((stage + 1u)); ++substage) {
          unsigned int half_net = (1u << (stage - substage));
          unsigned int partner = (tid ^ half_net);
          unsigned int direction_bit = ((tid >> (stage + 1u)) & 1u);
          if ((partner < 128u)) {
            float my_dist = shared_dists[tid];
            float partner_d = shared_dists[partner];
            unsigned int should_swap = (((tid < partner)) ? ((((direction_bit == 0u)) ? ((((my_dist > partner_d)) ? (1u) : (0u))) : ((((my_dist < partner_d)) ? (1u) : (0u))))) : (0u));
            if ((should_swap == 1u)) {
              unsigned int my_id = shared_ids[tid];
              unsigned int partner_id = shared_ids[partner];
              shared_ids[tid] = partner_id;
              shared_ids[partner] = my_id;
              shared_dists[tid] = partner_d;
              shared_dists[partner] = my_dist;
            }
          }
          __syncthreads();
        }
      }
      if ((tid < cmd)) {
        unsigned int write_idx = ((vid * cmd) + tid);
        if ((tid < tmd)) {
          unsigned int sorted_id = shared_ids[tid];
          graph[write_idx] = sorted_id;
        } else {
          graph[write_idx] = inv;
        }
      }
      if ((tid == 0u)) {
        degrees[vid] = tmd;
      }
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_accumulate_p_q_codebook_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_accumulate_p_q_codebook_i32(const float* subvectors, const unsigned int* assignments, const unsigned int* n_vectors, const unsigned int* sub_dim, int* codebook_sums_i32, unsigned int* codebook_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int sd = sub_dim[0];
  float scale = __uint_as_float(0x47800000u);
  if ((vid < nv)) {
    unsigned int cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      unsigned int src_idx = ((vid * sd) + k);
      unsigned int dst_idx = ((cw * sd) + k);
      float val_f32 = subvectors[src_idx];
      float scaled_f32 = (val_f32 * scale);
      int scaled_i32 = ((int)(scaled_f32));
      int _old = atomicAdd(&(codebook_sums_i32[dst_idx]), scaled_i32);
    }
    unsigned int _cnt = atomicAdd(&(codebook_counts[cw]), 1u);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_normalize_p_q_codebook_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_normalize_p_q_codebook_f32(const int* codebook_sums_i32, const unsigned int* codebook_counts, const unsigned int* n_codewords, const unsigned int* sub_dim, float* codebook) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nc = n_codewords[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nc * sd);
  float scale = __uint_as_float(0x47800000u);
  if ((tid < total)) {
    unsigned int cw = (tid / sd);
    unsigned int count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > __uint_as_float(0x00000000u))) ? ((sum_f32 / count_f32)) : (__uint_as_float(0x00000000u)));
    codebook[tid] = _centroid;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_train_p_q_codebook_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_train_p_q_codebook_f32(const float* subvectors, const unsigned int* assignments, const unsigned int* n_vectors, const unsigned int* n_codewords, const unsigned int* sub_dim, float* codebook_sums, unsigned int* codebook_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int sd = sub_dim[0];
  if ((vid < nv)) {
    unsigned int cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      unsigned int src_idx = ((vid * sd) + k);
      unsigned int dst_idx = ((cw * sd) + k);
      float val = subvectors[src_idx];
      float old = codebook_sums[dst_idx];
      float new_val = (old + val);
      codebook_sums[dst_idx] = new_val;
    }
    unsigned int _cnt = atomicAdd(&(codebook_counts[cw]), 1u);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_l2_squared_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_l2_squared_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    unsigned int dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_inner_product_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_inner_product_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float product = (x_val * y_val);
      float old_acc = acc;
      acc = (old_acc + product);
    }
    unsigned int dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_l1_distance_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_l1_distance_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float neg_diff = (__uint_as_float(0x00000000u) - diff);
      bool is_neg = (diff < __uint_as_float(0x00000000u));
      float abs_diff = ((is_neg) ? (neg_diff) : (diff));
      float old_acc = acc;
      acc = (old_acc + abs_diff);
    }
    unsigned int dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_batch_l2_to_neighbors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_batch_l2_to_neighbors_f32(const float* query, const float* dataset, const unsigned int* indices, const unsigned int* n_neighbors, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = n_neighbors[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int vec_idx = indices[gid];
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      unsigned int d_idx = ((vec_idx * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    float result = acc;
    distances[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_l2_squared_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_vector_search_l2_squared_f16(const __half* x, const __half* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      __half x_f16 = x[x_idx];
      __half y_f16 = y[y_idx];
      float x_val = ((float)(x_f16));
      float y_val = ((float)(y_f16));
      float diff = (x_val - y_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    unsigned int dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_cosine_similarity_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_cosine_similarity_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* sim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float dot = __uint_as_float(0x00000000u);
    float norm_x = __uint_as_float(0x00000000u);
    float norm_y = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float prod = (x_val * y_val);
      float old_dot = dot;
      dot = (old_dot + prod);
      float x_sq = (x_val * x_val);
      float y_sq = (y_val * y_val);
      float old_nx = norm_x;
      float old_ny = norm_y;
      norm_x = (old_nx + x_sq);
      norm_y = (old_ny + y_sq);
    }
    float final_dot = dot;
    float final_nx = norm_x;
    float final_ny = norm_y;
    float sqrt_nx = sqrt(final_nx);
    float sqrt_ny = sqrt(final_ny);
    float denom = (sqrt_nx * sqrt_ny);
    float eps = __uint_as_float(0x322bcc77u);
    float safe_denom = (denom + eps);
    float cosine = (final_dot / safe_denom);
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    sim[out_idx] = cosine;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_linf_distance_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_linf_distance_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float max_diff = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float abs_diff = abs(diff);
      float cur_max = max_diff;
      bool new_max = (abs_diff > cur_max);
      float updated_max = ((new_max) ? (abs_diff) : (cur_max));
      max_diff = updated_max;
    }
    float result = max_diff;
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_hamming_distance_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_hamming_distance_u32(const unsigned int* x, const unsigned int* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim_words, unsigned int* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    unsigned int count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      unsigned int x_idx = ((gid_x * dw) + k);
      unsigned int y_idx = ((gid_y * dw) + k);
      unsigned int x_val = x[x_idx];
      unsigned int y_val = y[y_idx];
      unsigned int xor_val = (x_val ^ y_val);
      unsigned int v0 = xor_val;
      unsigned int m1 = 1431655765u;
      unsigned int m2 = 858993459u;
      unsigned int m4 = 252645135u;
      unsigned int v1 = (v0 - ((v0 >> 1u) & m1));
      unsigned int v2 = ((v1 & m2) + ((v1 >> 2u) & m2));
      unsigned int v3 = ((v2 + (v2 >> 4u)) & m4);
      unsigned int v4 = (v3 + (v3 >> 8u));
      unsigned int v5 = (v4 + (v4 >> 16u));
      unsigned int popcount = (v5 & 63u);
      unsigned int old_count = count;
      count = (old_count + popcount);
    }
    unsigned int result = count;
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_correlation_distance_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_correlation_distance_f32(const float* x, const float* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float sum_x = __uint_as_float(0x00000000u);
    float sum_y = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float old_sx = sum_x;
      float old_sy = sum_y;
      sum_x = (old_sx + x_val);
      sum_y = (old_sy + y_val);
    }
    float d_f = ((float)(d));
    float final_sx = sum_x;
    float final_sy = sum_y;
    float mean_x = (final_sx / d_f);
    float mean_y = (final_sy / d_f);
    float cov = __uint_as_float(0x00000000u);
    float var_x = __uint_as_float(0x00000000u);
    float var_y = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx2 = ((gid_x * d) + k);
      unsigned int y_idx2 = ((gid_y * d) + k);
      float x_val2 = x[x_idx2];
      float y_val2 = y[y_idx2];
      float dx = (x_val2 - mean_x);
      float dy = (y_val2 - mean_y);
      float old_cov = cov;
      float old_vx = var_x;
      float old_vy = var_y;
      cov = (old_cov + (dx * dy));
      var_x = (old_vx + (dx * dx));
      var_y = (old_vy + (dy * dy));
    }
    float final_cov = cov;
    float final_vx = var_x;
    float final_vy = var_y;
    float var_prod = (final_vx * final_vy);
    float sqrt_var = sqrt(var_prod);
    float eps = __uint_as_float(0x322bcc77u);
    float safe_denom = (sqrt_var + eps);
    float correlation = (final_cov / safe_denom);
    float corr_dist = (__uint_as_float(0x3f800000u) - correlation);
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = corr_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_jaccard_distance_u32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_jaccard_distance_u32(const unsigned int* x, const unsigned int* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim_words, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    unsigned int intersection = 0u;
    unsigned int union_count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      unsigned int x_idx = ((gid_x * dw) + k);
      unsigned int y_idx = ((gid_y * dw) + k);
      unsigned int x_val = x[x_idx];
      unsigned int y_val = y[y_idx];
      unsigned int inter = (x_val & y_val);
      unsigned int uni = (x_val | y_val);
      unsigned int i0 = inter;
      unsigned int m1 = 1431655765u;
      unsigned int m2 = 858993459u;
      unsigned int m4 = 252645135u;
      unsigned int i1 = (i0 - ((i0 >> 1u) & m1));
      unsigned int i2 = ((i1 & m2) + ((i1 >> 2u) & m2));
      unsigned int i3 = ((i2 + (i2 >> 4u)) & m4);
      unsigned int i4 = (i3 + (i3 >> 8u));
      unsigned int i5 = (i4 + (i4 >> 16u));
      unsigned int pop_inter = (i5 & 63u);
      unsigned int u0 = uni;
      unsigned int u1 = (u0 - ((u0 >> 1u) & m1));
      unsigned int u2 = ((u1 & m2) + ((u1 >> 2u) & m2));
      unsigned int u3 = ((u2 + (u2 >> 4u)) & m4);
      unsigned int u4 = (u3 + (u3 >> 8u));
      unsigned int u5 = (u4 + (u4 >> 16u));
      unsigned int pop_union = (u5 & 63u);
      unsigned int old_inter = intersection;
      unsigned int old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    unsigned int final_inter = intersection;
    unsigned int final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = __uint_as_float(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (__uint_as_float(0x3f800000u) - similarity);
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_assign_labels_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_assign_labels_f32(const float* data, const float* centroids, const unsigned int* n_points, const unsigned int* k, const unsigned int* dim, unsigned int* labels, float* min_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = n_points[0];
  unsigned int kVal = k[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float best_dist = __uint_as_float(0x7e967699u);
    unsigned int best_label = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      float dist_acc = __uint_as_float(0x00000000u);
      for (int j = 0; j < (int)(d); ++j) {
        unsigned int data_idx = ((gid * d) + j);
        float data_val = data[data_idx];
        unsigned int cent_idx = ((c * d) + j);
        float cent_val = centroids[cent_idx];
        float diff = (data_val - cent_val);
        float diff_sq = (diff * diff);
        float old_acc = dist_acc;
        dist_acc = (old_acc + diff_sq);
      }
      float curr_dist = dist_acc;
      float curr_best = best_dist;
      bool is_better = (curr_dist < curr_best);
      if (is_better) {
        best_dist = curr_dist;
        best_label = c;
      }
    }
    unsigned int final_label = best_label;
    float final_dist = best_dist;
    labels[gid] = final_label;
    min_distances[gid] = final_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_accumulate_cluster_sums_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_accumulate_cluster_sums_f32(const float* data, const unsigned int* labels, const unsigned int* n_points, const unsigned int* k, const unsigned int* dim, float* cluster_sums, unsigned int* cluster_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = n_points[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int label = labels[gid];
    unsigned int _old_count = atomicAdd(&(cluster_counts[label]), 1u);
    for (int j = 0; j < (int)(d); ++j) {
      unsigned int data_idx = ((gid * d) + j);
      float data_val = data[data_idx];
      unsigned int sum_idx = ((label * d) + j);
      atomicAdd(&cluster_sums[sum_idx], data_val);
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_update_centroids_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_update_centroids_f32(const float* cluster_sums, const unsigned int* cluster_counts, const unsigned int* k, const unsigned int* dim, float* centroids) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  unsigned int d = dim[0];
  unsigned int total = (kVal * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int cluster_idx = (gid / d);
    unsigned int dim_idx = (gid % d);
    unsigned int count = cluster_counts[cluster_idx];
    float sum_val = cluster_sums[gid];
    float count_f = ((float)(count));
    bool is_empty = (count == 0u);
    float safe_count = ((is_empty) ? (__uint_as_float(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_clear_cluster_sums_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_clear_cluster_sums_f32(const unsigned int* total, float* cluster_sums) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = total[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    cluster_sums[gid] = __uint_as_float(0x00000000u);
  }
}

)kernel";
const char* k_hip_navatala_vector_search_clear_cluster_counts = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_clear_cluster_counts(const unsigned int* k, unsigned int* cluster_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    cluster_counts[gid] = 0u;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_reduce_inertia_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_reduce_inertia_f32(const float* min_distances, const unsigned int* n_points, float* partial_sums) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int group_id = ((unsigned int)((int)(blockIdx.x)));
  unsigned int n = n_points[0];
  __shared__ float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (__uint_as_float(0x00000000u)));
  sdata[lid] = val;
  __syncthreads();
  unsigned int stride = 128u;
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!((stride > 0u))) break;
    unsigned int s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      unsigned int other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    __syncthreads();
    stride = (s / 2u);
  }
  bool is_thread0 = (lid == 0u);
  if (is_thread0) {
    float block_sum = sdata[0u];
    partial_sums[group_id] = block_sum;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_bitonic_sort_step_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_bitonic_sort_step_f32(const unsigned int* n, const unsigned int* step, const unsigned int* stage, float* keys, unsigned int* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int stepVal = step[0];
  unsigned int stageVal = stage[0];
  unsigned int half_n = (nVal / 2u);
  bool inBounds = (gid < half_n);
  if (inBounds) {
    unsigned int distance = (1u << (stepVal - stageVal));
    unsigned int block_size = (2u << stepVal);
    unsigned int block_id = (gid / distance);
    unsigned int offset = (gid % distance);
    unsigned int i = ((block_id * (2u * distance)) + offset);
    unsigned int j = (i + distance);
    unsigned int block_start = ((i / block_size) * block_size);
    bool ascending = (((block_start / block_size) % 2u) == 0u);
    float key_i = keys[i];
    float key_j = keys[j];
    unsigned int val_i = values[i];
    unsigned int val_j = values[j];
    bool should_swap_asc = (key_i > key_j);
    bool should_swap_desc = (key_i < key_j);
    bool should_swap = ((ascending) ? (should_swap_asc) : (should_swap_desc));
    if (should_swap) {
      keys[i] = key_j;
      keys[j] = key_i;
      values[i] = val_j;
      values[j] = val_i;
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_per_row_top_k_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_per_row_top_k_f32(const float* data, const unsigned int* n_rows, const unsigned int* row_size, const unsigned int* k, float* top_values, unsigned int* top_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nr = n_rows[0];
  unsigned int rs = row_size[0];
  unsigned int kVal = k[0];
  unsigned int total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row_idx = (gid / kVal);
    unsigned int k_idx = (gid % kVal);
    unsigned int row_start = (row_idx * rs);
    float best_val = __uint_as_float(0x7e967699u);
    unsigned int best_idx = 0u;
    unsigned int count_smaller = 0u;
    for (int i = 0; i < (int)(rs); ++i) {
      unsigned int data_idx = (row_start + i);
      float val = data[data_idx];
      unsigned int smaller_count = 0u;
      for (int j = 0; j < (int)(rs); ++j) {
        unsigned int other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        unsigned int old_cnt = smaller_count;
        unsigned int incr = ((counts) ? (1u) : (0u));
        smaller_count = (old_cnt + incr);
      }
      unsigned int cnt = smaller_count;
      bool is_kth = (cnt == k_idx);
      if (is_kth) {
        best_val = val;
        best_idx = i;
      }
    }
    unsigned int out_idx = ((row_idx * kVal) + k_idx);
    float final_val = best_val;
    unsigned int final_idx = best_idx;
    top_values[out_idx] = final_val;
    top_indices[out_idx] = final_idx;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_merge_sorted_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_merge_sorted_f32(const float* a, const float* b, const unsigned int* a_indices, const unsigned int* b_indices, const unsigned int* n_a, const unsigned int* n_b, float* merged, unsigned int* merged_indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int na = n_a[0];
  unsigned int nb = n_b[0];
  unsigned int total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int lo = 0u;
    unsigned int hi = gid;
    unsigned int hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!((lo < hi))) break;
      unsigned int lo_v = lo;
      unsigned int hi_v = hi;
      unsigned int mid = (lo_v + ((hi_v - lo_v) / 2u));
      unsigned int j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (__uint_as_float(0x7e967699u)));
      float b_prev = (((j > 0u)) ? (b[(j - 1u)]) : (__uint_as_float(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + 1u);
      } else {
        hi = mid;
      }
    }
    unsigned int i = lo;
    unsigned int j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (__uint_as_float(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (__uint_as_float(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    unsigned int out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_rerank_with_exact_distances_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_rerank_with_exact_distances_f32(const float* queries, const float* dataset, const unsigned int* candidate_ids, const unsigned int* n_queries, const unsigned int* k, const unsigned int* dim, float* exact_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  unsigned int d = dim[0];
  unsigned int total = (nq * k_val);
  if ((gid < total)) {
    unsigned int query_id = (gid / k_val);
    unsigned int k_idx = (gid % k_val);
    unsigned int cand_id = candidate_ids[gid];
    float dist = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(d); ++i) {
      unsigned int q_idx = ((query_id * d) + i);
      unsigned int c_idx = ((cand_id * d) + i);
      float q_val = queries[q_idx];
      float c_val = dataset[c_idx];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_dist = dist;
      dist = (old_dist + sq);
    }
    float final_dist = dist;
    exact_distances[gid] = final_dist;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_filter_by_threshold_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_filter_by_threshold_f32(const float* distances, unsigned int* candidate_ids, const unsigned int* n_queries, const unsigned int* k, const float* threshold, const unsigned int* invalid_id, unsigned int* filtered_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  float thresh = threshold[0];
  unsigned int inv = invalid_id[0];
  if ((query_id < nq)) {
    unsigned int count = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      unsigned int idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if ((dist > thresh)) {
        candidate_ids[idx] = inv;
      } else {
        unsigned int old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    unsigned int final_count = count;
    filtered_counts[query_id] = final_count;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_merge_search_results_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_merge_search_results_f32(const unsigned int* ids_a, const float* dists_a, const unsigned int* ids_b, const float* dists_b, const unsigned int* n_queries, const unsigned int* k_a, const unsigned int* k_b, const unsigned int* k_out, unsigned int* merged_ids, float* merged_dists) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int ka = k_a[0];
  unsigned int kb = k_b[0];
  unsigned int ko = k_out[0];
  if ((query_id < nq)) {
    unsigned int ptr_a = 0u;
    unsigned int ptr_b = 0u;
    unsigned int out_idx = 0u;
    for (int _iter = 0; _iter < (int)(ko); ++_iter) {
      unsigned int pa = ptr_a;
      unsigned int pb = ptr_b;
      unsigned int oi = out_idx;
      bool a_valid = (pa < ka);
      bool b_valid = (pb < kb);
      unsigned int idx_a = ((query_id * ka) + pa);
      unsigned int idx_b = ((query_id * kb) + pb);
      float dist_a_raw = dists_a[idx_a];
      float dist_b_raw = dists_b[idx_b];
      float dist_a = ((a_valid) ? (dist_a_raw) : (__uint_as_float(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (__uint_as_float(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      unsigned int chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      unsigned int out_flat = ((query_id * ko) + oi);
      merged_ids[out_flat] = chosen_id;
      merged_dists[out_flat] = chosen_dist;
      if (choose_a) {
        ptr_a = (pa + 1u);
      } else {
        ptr_b = (pb + 1u);
      }
      out_idx = (oi + 1u);
    }
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_recall_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_recall_f32(const unsigned int* approx_ids, const unsigned int* ground_truth_ids, const unsigned int* n_queries, const unsigned int* k, float* recall_per_query) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int query_id = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int k_val = k[0];
  if ((query_id < nq)) {
    unsigned int hits = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      unsigned int approx_idx = ((query_id * k_val) + i);
      unsigned int approx_id = approx_ids[approx_idx];
      unsigned int found = 0u;
      for (int j = 0; j < (int)(k_val); ++j) {
        unsigned int gt_idx = ((query_id * k_val) + j);
        unsigned int gt_id = ground_truth_ids[gt_idx];
        if ((approx_id == gt_id)) {
          found = 1u;
        }
      }
      unsigned int was_found = found;
      unsigned int old_hits = hits;
      hits = (old_hits + was_found);
    }
    unsigned int total_hits = hits;
    float hits_f32 = ((float)(total_hits));
    float k_f32 = ((float)(k_val));
    float recall = (hits_f32 / k_f32);
    recall_per_query[query_id] = recall;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_scalar_quantize_f32_to_i8 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_scalar_quantize_f32_to_i8(const float* _input, const unsigned int* n_vectors, const unsigned int* dim, const float* mins, const float* maxs, signed char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  if ((gid < total)) {
    unsigned int dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * __uint_as_float(0x437f0000u));
    float offset = (scaled - __uint_as_float(0x43000000u));
    float neg128 = (__uint_as_float(0x00000000u) - __uint_as_float(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > __uint_as_float(0x42fe0000u))) ? (__uint_as_float(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    signed char quantized = ((signed char)(quantized_i32));
    _output[gid] = quantized;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_scalar_dequantize_i8_to_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_scalar_dequantize_i8_to_f32(const signed char* _input, const unsigned int* n_vectors, const unsigned int* dim, const float* mins, const float* maxs, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  if ((gid < total)) {
    unsigned int dim_idx = (gid % d);
    signed char qval = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float qval_f32 = ((float)(qval));
    float offset = (qval_f32 + __uint_as_float(0x43000000u));
    float normalized = (offset / __uint_as_float(0x437f0000u));
    float range = (max_val - min_val);
    float dequant = ((normalized * range) + min_val);
    _output[gid] = dequant;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_compute_quantized_distances_i8 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_compute_quantized_distances_i8(const signed char* queries, const signed char* dataset, const unsigned int* n_queries, const unsigned int* n_dataset, const unsigned int* dim, const float* scale, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nq = n_queries[0];
  unsigned int nd = n_dataset[0];
  unsigned int d = dim[0];
  float s = scale[0];
  bool in_x = (gid_x < nq);
  bool in_y = (gid_y < nd);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    int acc = 0;
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((gid_x * d) + k);
      unsigned int d_idx = ((gid_y * d) + k);
      signed char q_val = queries[q_idx];
      signed char d_val = dataset[d_idx];
      int q_i32 = ((int)(q_val));
      int d_i32 = ((int)(d_val));
      int diff = (q_i32 - d_i32);
      int sq = (diff * diff);
      int old_acc = acc;
      acc = (old_acc + sq);
    }
    int final_acc = acc;
    float acc_f32 = ((float)(final_acc));
    float result = (acc_f32 * s);
    unsigned int out_idx = ((gid_x * nd) + gid_y);
    distances[out_idx] = result;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_encode_p_q_vectors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_encode_p_q_vectors_f32(const float* subvectors, const float* codebook, const unsigned int* n_vectors, const unsigned int* n_codewords, const unsigned int* sub_dim, unsigned char* codes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int nc = n_codewords[0];
  unsigned int sd = sub_dim[0];
  if ((vid < nv)) {
    unsigned int best_cw = 0u;
    float best_dist = __uint_as_float(0x7e967699u);
    for (int cw = 0; cw < (int)(nc); ++cw) {
      float dist = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        unsigned int sv_idx = ((vid * sd) + k);
        unsigned int cb_idx = ((cw * sd) + k);
        float sv = subvectors[sv_idx];
        float cb = codebook[cb_idx];
        float diff = (sv - cb);
        float sq = (diff * diff);
        float old_dist = dist;
        dist = (old_dist + sq);
      }
      float cw_dist = dist;
      float curr_best = best_dist;
      if ((cw_dist < curr_best)) {
        best_cw = cw;
        best_dist = cw_dist;
      }
    }
    unsigned int final_cw = best_cw;
    unsigned char code = ((unsigned char)(final_cw));
    codes[vid] = code;
  }
}

)kernel";
const char* k_hip_navatala_vector_search_decode_p_q_vectors_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_decode_p_q_vectors_f32(const unsigned char* codes, const float* codebook, const unsigned int* n_vectors, const unsigned int* sub_dim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nv * sd);
  if ((gid < total)) {
    unsigned int vid = (gid / sd);
    unsigned int k = (gid % sd);
    unsigned char code = codes[vid];
    unsigned int cw = ((unsigned int)(code));
    unsigned int cb_idx = ((cw * sd) + k);
    float val = codebook[cb_idx];
    _output[gid] = val;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_init_seeds_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "seed_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_seeds", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "itopk_size", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_init_seeds_f32 = {
  1,
  "navatala_vector_search_init_seeds_f32",
  "hip",
  "navatala_vector_search_init_seeds_f32",
  "kernel:hip:navatala_vector_search_init_seeds_f32",
  "abi-r1:hip:navatala_vector_search_init_seeds_f32",
  "abi-r1:hip:navatala_vector_search_init_seeds_f32",
  9,
  kAbiArgs_hip_navatala_vector_search_init_seeds_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_expand_neighbors_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "top_candidates", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "search_width", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "new_candidates", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_distances", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_expand_neighbors_f32 = {
  1,
  "navatala_vector_search_expand_neighbors_f32",
  "hip",
  "navatala_vector_search_expand_neighbors_f32",
  "kernel:hip:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_expand_neighbors_f32",
  10,
  kAbiArgs_hip_navatala_vector_search_expand_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_check_visited[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "visited", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hash_size", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "is_new", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_check_visited = {
  1,
  "navatala_vector_search_check_visited",
  "hip",
  "navatala_vector_search_check_visited",
  "kernel:hip:navatala_vector_search_check_visited",
  "abi-r1:hip:navatala_vector_search_check_visited",
  "abi-r1:hip:navatala_vector_search_check_visited",
  5,
  kAbiArgs_hip_navatala_vector_search_check_visited
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_clear_visited[] = {
  { "hash_size", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_clear_visited = {
  1,
  "navatala_vector_search_clear_visited",
  "hip",
  "navatala_vector_search_clear_visited",
  "kernel:hip:navatala_vector_search_clear_visited",
  "abi-r1:hip:navatala_vector_search_clear_visited",
  "abi-r1:hip:navatala_vector_search_clear_visited",
  2,
  kAbiArgs_hip_navatala_vector_search_clear_visited
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_extract_results_f32[] = {
  { "sorted_candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sorted_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "itopk_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_extract_results_f32 = {
  1,
  "navatala_vector_search_extract_results_f32",
  "hip",
  "navatala_vector_search_extract_results_f32",
  "kernel:hip:navatala_vector_search_extract_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_results_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_extract_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_distances_batch_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_distances_batch_f32 = {
  1,
  "navatala_vector_search_compute_distances_batch_f32",
  "hip",
  "navatala_vector_search_compute_distances_batch_f32",
  "kernel:hip:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:hip:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:hip:navatala_vector_search_compute_distances_batch_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_compute_distances_batch_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_search_layer_greedy_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entry_points", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ef", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_distances", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_found", 11, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_search_layer_greedy_f32 = {
  1,
  "navatala_vector_search_search_layer_greedy_f32",
  "hip",
  "navatala_vector_search_search_layer_greedy_f32",
  "kernel:hip:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:hip:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:hip:navatala_vector_search_search_layer_greedy_f32",
  12,
  kAbiArgs_hip_navatala_vector_search_search_layer_greedy_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_select_neighbors_simple_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selected", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "selected_count", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_select_neighbors_simple_f32 = {
  1,
  "navatala_vector_search_select_neighbors_simple_f32",
  "hip",
  "navatala_vector_search_select_neighbors_simple_f32",
  "kernel:hip:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:hip:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:hip:navatala_vector_search_select_neighbors_simple_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_select_neighbors_simple_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_select_neighbors_heuristic_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selected", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_selected", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_select_neighbors_heuristic_f32 = {
  1,
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "hip",
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "kernel:hip:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:hip:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:hip:navatala_vector_search_select_neighbors_heuristic_f32",
  9,
  kAbiArgs_hip_navatala_vector_search_select_neighbors_heuristic_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_update_candidate_list_f32[] = {
  { "candidates", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_distances", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_current", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_new", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ef", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_merged", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_update_candidate_list_f32 = {
  1,
  "navatala_vector_search_update_candidate_list_f32",
  "hip",
  "navatala_vector_search_update_candidate_list_f32",
  "kernel:hip:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:hip:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:hip:navatala_vector_search_update_candidate_list_f32",
  8,
  kAbiArgs_hip_navatala_vector_search_update_candidate_list_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_extract_layer_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_found", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_extract_layer_results_f32 = {
  1,
  "navatala_vector_search_extract_layer_results_f32",
  "hip",
  "navatala_vector_search_extract_layer_results_f32",
  "kernel:hip:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_layer_results_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_extract_layer_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_init_search_state_f32[] = {
  { "global_entry", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_queries", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "entry_points", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_init_search_state_f32 = {
  1,
  "navatala_vector_search_init_search_state_f32",
  "hip",
  "navatala_vector_search_init_search_state_f32",
  "kernel:hip:navatala_vector_search_init_search_state_f32",
  "abi-r1:hip:navatala_vector_search_init_search_state_f32",
  "abi-r1:hip:navatala_vector_search_init_search_state_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_init_search_state_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_mark_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_mark_visited_batch = {
  1,
  "navatala_vector_search_mark_visited_batch",
  "hip",
  "navatala_vector_search_mark_visited_batch",
  "kernel:hip:navatala_vector_search_mark_visited_batch",
  "abi-r1:hip:navatala_vector_search_mark_visited_batch",
  "abi-r1:hip:navatala_vector_search_mark_visited_batch",
  5,
  kAbiArgs_hip_navatala_vector_search_mark_visited_batch
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_check_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "is_visited", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_check_visited_batch = {
  1,
  "navatala_vector_search_check_visited_batch",
  "hip",
  "navatala_vector_search_check_visited_batch",
  "kernel:hip:navatala_vector_search_check_visited_batch",
  "abi-r1:hip:navatala_vector_search_check_visited_batch",
  "abi-r1:hip:navatala_vector_search_check_visited_batch",
  6,
  kAbiArgs_hip_navatala_vector_search_check_visited_batch
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_merge_multi_query_results_f32[] = {
  { "layer_results", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "layer_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_layers", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "final_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "final_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_merge_multi_query_results_f32 = {
  1,
  "navatala_vector_search_merge_multi_query_results_f32",
  "hip",
  "navatala_vector_search_merge_multi_query_results_f32",
  "kernel:hip:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_multi_query_results_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_merge_multi_query_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_neighbor_distances_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_neighbor_distances_f32 = {
  1,
  "navatala_vector_search_compute_neighbor_distances_f32",
  "hip",
  "navatala_vector_search_compute_neighbor_distances_f32",
  "kernel:hip:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_neighbor_distances_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_compute_neighbor_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_greedy_search_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entry_point", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_queries", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "search_list_size", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_dists", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 11, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_greedy_search_f32 = {
  1,
  "navatala_vector_search_greedy_search_f32",
  "hip",
  "navatala_vector_search_greedy_search_f32",
  "kernel:hip:navatala_vector_search_greedy_search_f32",
  "abi-r1:hip:navatala_vector_search_greedy_search_f32",
  "abi-r1:hip:navatala_vector_search_greedy_search_f32",
  12,
  kAbiArgs_hip_navatala_vector_search_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_robust_prune_f32[] = {
  { "vertex", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "alpha", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selected", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_selected", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_robust_prune_f32 = {
  1,
  "navatala_vector_search_robust_prune_f32",
  "hip",
  "navatala_vector_search_robust_prune_f32",
  "kernel:hip:navatala_vector_search_robust_prune_f32",
  "abi-r1:hip:navatala_vector_search_robust_prune_f32",
  "abi-r1:hip:navatala_vector_search_robust_prune_f32",
  10,
  kAbiArgs_hip_navatala_vector_search_robust_prune_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_insert_vertex_f32[] = {
  { "vertex_id", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_insert_vertex_f32 = {
  1,
  "navatala_vector_search_insert_vertex_f32",
  "hip",
  "navatala_vector_search_insert_vertex_f32",
  "kernel:hip:navatala_vector_search_insert_vertex_f32",
  "abi-r1:hip:navatala_vector_search_insert_vertex_f32",
  "abi-r1:hip:navatala_vector_search_insert_vertex_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_insert_vertex_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_update_neighbors_f32[] = {
  { "new_vertex", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "new_neighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_new_neighbors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_update_neighbors_f32 = {
  1,
  "navatala_vector_search_update_neighbors_f32",
  "hip",
  "navatala_vector_search_update_neighbors_f32",
  "kernel:hip:navatala_vector_search_update_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_update_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_update_neighbors_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_update_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_random_init_neighbors_f32[] = {
  { "n_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "init_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_random_init_neighbors_f32 = {
  1,
  "navatala_vector_search_random_init_neighbors_f32",
  "hip",
  "navatala_vector_search_random_init_neighbors_f32",
  "kernel:hip:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_random_init_neighbors_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_random_init_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_extract_search_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_dists", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "search_list_size", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_extract_search_results_f32 = {
  1,
  "navatala_vector_search_extract_search_results_f32",
  "hip",
  "navatala_vector_search_extract_search_results_f32",
  "kernel:hip:navatala_vector_search_extract_search_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_search_results_f32",
  "abi-r1:hip:navatala_vector_search_extract_search_results_f32",
  8,
  kAbiArgs_hip_navatala_vector_search_extract_search_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_batched_greedy_search_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entry_point", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_queries", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_vectors", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_batched_greedy_search_f32 = {
  1,
  "navatala_vector_search_batched_greedy_search_f32",
  "hip",
  "navatala_vector_search_batched_greedy_search_f32",
  "kernel:hip:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:hip:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:hip:navatala_vector_search_batched_greedy_search_f32",
  12,
  kAbiArgs_hip_navatala_vector_search_batched_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_centroid_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroid_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_centroid_distances_f32 = {
  1,
  "navatala_vector_search_compute_centroid_distances_f32",
  "hip",
  "navatala_vector_search_compute_centroid_distances_f32",
  "kernel:hip:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_centroid_distances_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_compute_centroid_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_select_probes_f32[] = {
  { "centroid_distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_probes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "probe_indices", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "probe_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_select_probes_f32 = {
  1,
  "navatala_vector_search_select_probes_f32",
  "hip",
  "navatala_vector_search_select_probes_f32",
  "kernel:hip:navatala_vector_search_select_probes_f32",
  "abi-r1:hip:navatala_vector_search_select_probes_f32",
  "abi-r1:hip:navatala_vector_search_select_probes_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_select_probes_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_scan_inverted_list_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_data", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "probe_indices", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_probes", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_list_size", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scan_distances", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scan_indices", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scan_counts", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_scan_inverted_list_f32 = {
  1,
  "navatala_vector_search_scan_inverted_list_f32",
  "hip",
  "navatala_vector_search_scan_inverted_list_f32",
  "kernel:hip:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:hip:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:hip:navatala_vector_search_scan_inverted_list_f32",
  12,
  kAbiArgs_hip_navatala_vector_search_scan_inverted_list_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_merge_probe_results_f32[] = {
  { "scan_distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scan_indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "scan_counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_probes", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_list_size", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "final_distances", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "final_indices", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_merge_probe_results_f32 = {
  1,
  "navatala_vector_search_merge_probe_results_f32",
  "hip",
  "navatala_vector_search_merge_probe_results_f32",
  "kernel:hip:navatala_vector_search_merge_probe_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_probe_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_probe_results_f32",
  9,
  kAbiArgs_hip_navatala_vector_search_merge_probe_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_assign_to_cluster_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_assign_to_cluster_f32 = {
  1,
  "navatala_vector_search_assign_to_cluster_f32",
  "hip",
  "navatala_vector_search_assign_to_cluster_f32",
  "kernel:hip:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:hip:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:hip:navatala_vector_search_assign_to_cluster_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_assign_to_cluster_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_count_cluster_sizes[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_counts", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_count_cluster_sizes = {
  1,
  "navatala_vector_search_count_cluster_sizes",
  "hip",
  "navatala_vector_search_count_cluster_sizes",
  "kernel:hip:navatala_vector_search_count_cluster_sizes",
  "abi-r1:hip:navatala_vector_search_count_cluster_sizes",
  "abi-r1:hip:navatala_vector_search_count_cluster_sizes",
  4,
  kAbiArgs_hip_navatala_vector_search_count_cluster_sizes
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_populate_inverted_lists_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "list_data", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_indices", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "write_positions", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_populate_inverted_lists_f32 = {
  1,
  "navatala_vector_search_populate_inverted_lists_f32",
  "hip",
  "navatala_vector_search_populate_inverted_lists_f32",
  "kernel:hip:navatala_vector_search_populate_inverted_lists_f32",
  "abi-r1:hip:navatala_vector_search_populate_inverted_lists_f32",
  "abi-r1:hip:navatala_vector_search_populate_inverted_lists_f32",
  8,
  kAbiArgs_hip_navatala_vector_search_populate_inverted_lists_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_initialize_random_graph[] = {
  { "n_nodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_initialize_random_graph = {
  1,
  "navatala_vector_search_initialize_random_graph",
  "hip",
  "navatala_vector_search_initialize_random_graph",
  "kernel:hip:navatala_vector_search_initialize_random_graph",
  "abi-r1:hip:navatala_vector_search_initialize_random_graph",
  "abi-r1:hip:navatala_vector_search_initialize_random_graph",
  5,
  kAbiArgs_hip_navatala_vector_search_initialize_random_graph
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_graph_distances_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_graph_distances_f32 = {
  1,
  "navatala_vector_search_compute_graph_distances_f32",
  "hip",
  "navatala_vector_search_compute_graph_distances_f32",
  "kernel:hip:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:hip:navatala_vector_search_compute_graph_distances_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_compute_graph_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_nn_descent_join_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "new_graph", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_distances", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "update_count", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_nn_descent_join_f32 = {
  1,
  "navatala_vector_search_nn_descent_join_f32",
  "hip",
  "navatala_vector_search_nn_descent_join_f32",
  "kernel:hip:navatala_vector_search_nn_descent_join_f32",
  "abi-r1:hip:navatala_vector_search_nn_descent_join_f32",
  "abi-r1:hip:navatala_vector_search_nn_descent_join_f32",
  9,
  kAbiArgs_hip_navatala_vector_search_nn_descent_join_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_build_reverse_graph[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "reverse_graph", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reverse_counts", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_build_reverse_graph = {
  1,
  "navatala_vector_search_build_reverse_graph",
  "hip",
  "navatala_vector_search_build_reverse_graph",
  "kernel:hip:navatala_vector_search_build_reverse_graph",
  "abi-r1:hip:navatala_vector_search_build_reverse_graph",
  "abi-r1:hip:navatala_vector_search_build_reverse_graph",
  5,
  kAbiArgs_hip_navatala_vector_search_build_reverse_graph
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_prune_graph_r_n_g_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pruned_graph", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_prune_graph_r_n_g_f32 = {
  1,
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "hip",
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "kernel:hip:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:hip:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:hip:navatala_vector_search_prune_graph_r_n_g_f32",
  8,
  kAbiArgs_hip_navatala_vector_search_prune_graph_r_n_g_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compact_pruned_graph[] = {
  { "pruned_graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "compact_graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compact_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compact_pruned_graph = {
  1,
  "navatala_vector_search_compact_pruned_graph",
  "hip",
  "navatala_vector_search_compact_pruned_graph",
  "kernel:hip:navatala_vector_search_compact_pruned_graph",
  "abi-r1:hip:navatala_vector_search_compact_pruned_graph",
  "abi-r1:hip:navatala_vector_search_compact_pruned_graph",
  7,
  kAbiArgs_hip_navatala_vector_search_compact_pruned_graph
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_vertex_degrees[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_vertex_degrees = {
  1,
  "navatala_vector_search_compute_vertex_degrees",
  "hip",
  "navatala_vector_search_compute_vertex_degrees",
  "kernel:hip:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:hip:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:hip:navatala_vector_search_compute_vertex_degrees",
  5,
  kAbiArgs_hip_navatala_vector_search_compute_vertex_degrees
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_prune_excess_degree[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_prune_excess_degree = {
  1,
  "navatala_vector_search_prune_excess_degree",
  "hip",
  "navatala_vector_search_prune_excess_degree",
  "kernel:hip:navatala_vector_search_prune_excess_degree",
  "abi-r1:hip:navatala_vector_search_prune_excess_degree",
  "abi-r1:hip:navatala_vector_search_prune_excess_degree",
  7,
  kAbiArgs_hip_navatala_vector_search_prune_excess_degree
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_reorder_vertices_by_degree[] = {
  { "degrees", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prefix_sums", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_bins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "old_to_new", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_to_old", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_reorder_vertices_by_degree = {
  1,
  "navatala_vector_search_reorder_vertices_by_degree",
  "hip",
  "navatala_vector_search_reorder_vertices_by_degree",
  "kernel:hip:navatala_vector_search_reorder_vertices_by_degree",
  "abi-r1:hip:navatala_vector_search_reorder_vertices_by_degree",
  "abi-r1:hip:navatala_vector_search_reorder_vertices_by_degree",
  6,
  kAbiArgs_hip_navatala_vector_search_reorder_vertices_by_degree
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compact_graph_edges[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compact_graph_edges = {
  1,
  "navatala_vector_search_compact_graph_edges",
  "hip",
  "navatala_vector_search_compact_graph_edges",
  "kernel:hip:navatala_vector_search_compact_graph_edges",
  "abi-r1:hip:navatala_vector_search_compact_graph_edges",
  "abi-r1:hip:navatala_vector_search_compact_graph_edges",
  5,
  kAbiArgs_hip_navatala_vector_search_compact_graph_edges
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_validate_graph_integrity[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "error_flags", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_validate_graph_integrity = {
  1,
  "navatala_vector_search_validate_graph_integrity",
  "hip",
  "navatala_vector_search_validate_graph_integrity",
  "kernel:hip:navatala_vector_search_validate_graph_integrity",
  "abi-r1:hip:navatala_vector_search_validate_graph_integrity",
  "abi-r1:hip:navatala_vector_search_validate_graph_integrity",
  5,
  kAbiArgs_hip_navatala_vector_search_validate_graph_integrity
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_prune_excess_degree_sorted[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_prune_excess_degree_sorted = {
  1,
  "navatala_vector_search_prune_excess_degree_sorted",
  "hip",
  "navatala_vector_search_prune_excess_degree_sorted",
  "kernel:hip:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:hip:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:hip:navatala_vector_search_prune_excess_degree_sorted",
  7,
  kAbiArgs_hip_navatala_vector_search_prune_excess_degree_sorted
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_accumulate_p_q_codebook_i32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook_sums_i32", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_accumulate_p_q_codebook_i32 = {
  1,
  "navatala_vector_search_accumulate_p_q_codebook_i32",
  "hip",
  "navatala_vector_search_accumulate_p_q_codebook_i32",
  "kernel:hip:navatala_vector_search_accumulate_p_q_codebook_i32",
  "abi-r1:hip:navatala_vector_search_accumulate_p_q_codebook_i32",
  "abi-r1:hip:navatala_vector_search_accumulate_p_q_codebook_i32",
  6,
  kAbiArgs_hip_navatala_vector_search_accumulate_p_q_codebook_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_normalize_p_q_codebook_f32[] = {
  { "codebook_sums_i32", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_codewords", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_normalize_p_q_codebook_f32 = {
  1,
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "hip",
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "kernel:hip:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:hip:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:hip:navatala_vector_search_normalize_p_q_codebook_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_normalize_p_q_codebook_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_train_p_q_codebook_f32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codewords", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook_sums", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_train_p_q_codebook_f32 = {
  1,
  "navatala_vector_search_train_p_q_codebook_f32",
  "hip",
  "navatala_vector_search_train_p_q_codebook_f32",
  "kernel:hip:navatala_vector_search_train_p_q_codebook_f32",
  "abi-r1:hip:navatala_vector_search_train_p_q_codebook_f32",
  "abi-r1:hip:navatala_vector_search_train_p_q_codebook_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_train_p_q_codebook_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_l2_squared_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_l2_squared_f32 = {
  1,
  "navatala_vector_search_l2_squared_f32",
  "hip",
  "navatala_vector_search_l2_squared_f32",
  "kernel:hip:navatala_vector_search_l2_squared_f32",
  "abi-r1:hip:navatala_vector_search_l2_squared_f32",
  "abi-r1:hip:navatala_vector_search_l2_squared_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_l2_squared_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_inner_product_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_inner_product_f32 = {
  1,
  "navatala_vector_search_inner_product_f32",
  "hip",
  "navatala_vector_search_inner_product_f32",
  "kernel:hip:navatala_vector_search_inner_product_f32",
  "abi-r1:hip:navatala_vector_search_inner_product_f32",
  "abi-r1:hip:navatala_vector_search_inner_product_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_l1_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_l1_distance_f32 = {
  1,
  "navatala_vector_search_l1_distance_f32",
  "hip",
  "navatala_vector_search_l1_distance_f32",
  "kernel:hip:navatala_vector_search_l1_distance_f32",
  "abi-r1:hip:navatala_vector_search_l1_distance_f32",
  "abi-r1:hip:navatala_vector_search_l1_distance_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_l1_distance_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_batch_l2_to_neighbors_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_batch_l2_to_neighbors_f32 = {
  1,
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "hip",
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "kernel:hip:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:hip:navatala_vector_search_batch_l2_to_neighbors_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_batch_l2_to_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_l2_squared_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_l2_squared_f16 = {
  1,
  "navatala_vector_search_l2_squared_f16",
  "hip",
  "navatala_vector_search_l2_squared_f16",
  "kernel:hip:navatala_vector_search_l2_squared_f16",
  "abi-r1:hip:navatala_vector_search_l2_squared_f16",
  "abi-r1:hip:navatala_vector_search_l2_squared_f16",
  6,
  kAbiArgs_hip_navatala_vector_search_l2_squared_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_cosine_similarity_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sim", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_cosine_similarity_f32 = {
  1,
  "navatala_vector_search_cosine_similarity_f32",
  "hip",
  "navatala_vector_search_cosine_similarity_f32",
  "kernel:hip:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:hip:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:hip:navatala_vector_search_cosine_similarity_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_cosine_similarity_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_linf_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_linf_distance_f32 = {
  1,
  "navatala_vector_search_linf_distance_f32",
  "hip",
  "navatala_vector_search_linf_distance_f32",
  "kernel:hip:navatala_vector_search_linf_distance_f32",
  "abi-r1:hip:navatala_vector_search_linf_distance_f32",
  "abi-r1:hip:navatala_vector_search_linf_distance_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_linf_distance_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_hamming_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_hamming_distance_u32 = {
  1,
  "navatala_vector_search_hamming_distance_u32",
  "hip",
  "navatala_vector_search_hamming_distance_u32",
  "kernel:hip:navatala_vector_search_hamming_distance_u32",
  "abi-r1:hip:navatala_vector_search_hamming_distance_u32",
  "abi-r1:hip:navatala_vector_search_hamming_distance_u32",
  6,
  kAbiArgs_hip_navatala_vector_search_hamming_distance_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_correlation_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_correlation_distance_f32 = {
  1,
  "navatala_vector_search_correlation_distance_f32",
  "hip",
  "navatala_vector_search_correlation_distance_f32",
  "kernel:hip:navatala_vector_search_correlation_distance_f32",
  "abi-r1:hip:navatala_vector_search_correlation_distance_f32",
  "abi-r1:hip:navatala_vector_search_correlation_distance_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_correlation_distance_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_jaccard_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_jaccard_distance_u32 = {
  1,
  "navatala_vector_search_jaccard_distance_u32",
  "hip",
  "navatala_vector_search_jaccard_distance_u32",
  "kernel:hip:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:hip:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:hip:navatala_vector_search_jaccard_distance_u32",
  6,
  kAbiArgs_hip_navatala_vector_search_jaccard_distance_u32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_assign_labels_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "min_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_assign_labels_f32 = {
  1,
  "navatala_vector_search_assign_labels_f32",
  "hip",
  "navatala_vector_search_assign_labels_f32",
  "kernel:hip:navatala_vector_search_assign_labels_f32",
  "abi-r1:hip:navatala_vector_search_assign_labels_f32",
  "abi-r1:hip:navatala_vector_search_assign_labels_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_assign_labels_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_accumulate_cluster_sums_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_sums", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_counts", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_accumulate_cluster_sums_f32 = {
  1,
  "navatala_vector_search_accumulate_cluster_sums_f32",
  "hip",
  "navatala_vector_search_accumulate_cluster_sums_f32",
  "kernel:hip:navatala_vector_search_accumulate_cluster_sums_f32",
  "abi-r1:hip:navatala_vector_search_accumulate_cluster_sums_f32",
  "abi-r1:hip:navatala_vector_search_accumulate_cluster_sums_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_accumulate_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_update_centroids_f32[] = {
  { "cluster_sums", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroids", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_update_centroids_f32 = {
  1,
  "navatala_vector_search_update_centroids_f32",
  "hip",
  "navatala_vector_search_update_centroids_f32",
  "kernel:hip:navatala_vector_search_update_centroids_f32",
  "abi-r1:hip:navatala_vector_search_update_centroids_f32",
  "abi-r1:hip:navatala_vector_search_update_centroids_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_update_centroids_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_clear_cluster_sums_f32[] = {
  { "total", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_sums", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_clear_cluster_sums_f32 = {
  1,
  "navatala_vector_search_clear_cluster_sums_f32",
  "hip",
  "navatala_vector_search_clear_cluster_sums_f32",
  "kernel:hip:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:hip:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:hip:navatala_vector_search_clear_cluster_sums_f32",
  2,
  kAbiArgs_hip_navatala_vector_search_clear_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_clear_cluster_counts[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_clear_cluster_counts = {
  1,
  "navatala_vector_search_clear_cluster_counts",
  "hip",
  "navatala_vector_search_clear_cluster_counts",
  "kernel:hip:navatala_vector_search_clear_cluster_counts",
  "abi-r1:hip:navatala_vector_search_clear_cluster_counts",
  "abi-r1:hip:navatala_vector_search_clear_cluster_counts",
  2,
  kAbiArgs_hip_navatala_vector_search_clear_cluster_counts
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_reduce_inertia_f32[] = {
  { "min_distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partial_sums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_reduce_inertia_f32 = {
  1,
  "navatala_vector_search_reduce_inertia_f32",
  "hip",
  "navatala_vector_search_reduce_inertia_f32",
  "kernel:hip:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:hip:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:hip:navatala_vector_search_reduce_inertia_f32",
  3,
  kAbiArgs_hip_navatala_vector_search_reduce_inertia_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_bitonic_sort_step_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "step", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stage", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_bitonic_sort_step_f32 = {
  1,
  "navatala_vector_search_bitonic_sort_step_f32",
  "hip",
  "navatala_vector_search_bitonic_sort_step_f32",
  "kernel:hip:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:hip:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:hip:navatala_vector_search_bitonic_sort_step_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_bitonic_sort_step_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_per_row_top_k_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "row_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "top_values", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "top_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_per_row_top_k_f32 = {
  1,
  "navatala_vector_search_per_row_top_k_f32",
  "hip",
  "navatala_vector_search_per_row_top_k_f32",
  "kernel:hip:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:hip:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:hip:navatala_vector_search_per_row_top_k_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_per_row_top_k_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_merge_sorted_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "a_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b_indices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_a", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_b", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "merged", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "merged_indices", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_merge_sorted_f32 = {
  1,
  "navatala_vector_search_merge_sorted_f32",
  "hip",
  "navatala_vector_search_merge_sorted_f32",
  "kernel:hip:navatala_vector_search_merge_sorted_f32",
  "abi-r1:hip:navatala_vector_search_merge_sorted_f32",
  "abi-r1:hip:navatala_vector_search_merge_sorted_f32",
  8,
  kAbiArgs_hip_navatala_vector_search_merge_sorted_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_rerank_with_exact_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "exact_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_rerank_with_exact_distances_f32 = {
  1,
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "hip",
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "kernel:hip:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:hip:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:hip:navatala_vector_search_rerank_with_exact_distances_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_rerank_with_exact_distances_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_filter_by_threshold_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "threshold", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "filtered_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_filter_by_threshold_f32 = {
  1,
  "navatala_vector_search_filter_by_threshold_f32",
  "hip",
  "navatala_vector_search_filter_by_threshold_f32",
  "kernel:hip:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:hip:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:hip:navatala_vector_search_filter_by_threshold_f32",
  7,
  kAbiArgs_hip_navatala_vector_search_filter_by_threshold_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_merge_search_results_f32[] = {
  { "ids_a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dists_a", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ids_b", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dists_b", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k_a", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k_b", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k_out", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "merged_ids", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "merged_dists", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_merge_search_results_f32 = {
  1,
  "navatala_vector_search_merge_search_results_f32",
  "hip",
  "navatala_vector_search_merge_search_results_f32",
  "kernel:hip:navatala_vector_search_merge_search_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_search_results_f32",
  "abi-r1:hip:navatala_vector_search_merge_search_results_f32",
  10,
  kAbiArgs_hip_navatala_vector_search_merge_search_results_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_recall_f32[] = {
  { "approx_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ground_truth_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recall_per_query", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_recall_f32 = {
  1,
  "navatala_vector_search_compute_recall_f32",
  "hip",
  "navatala_vector_search_compute_recall_f32",
  "kernel:hip:navatala_vector_search_compute_recall_f32",
  "abi-r1:hip:navatala_vector_search_compute_recall_f32",
  "abi-r1:hip:navatala_vector_search_compute_recall_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_compute_recall_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_scalar_quantize_f32_to_i8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_scalar_quantize_f32_to_i8 = {
  1,
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "hip",
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "kernel:hip:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:hip:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:hip:navatala_vector_search_scalar_quantize_f32_to_i8",
  6,
  kAbiArgs_hip_navatala_vector_search_scalar_quantize_f32_to_i8
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_scalar_dequantize_i8_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_scalar_dequantize_i8_to_f32 = {
  1,
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "hip",
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "kernel:hip:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:hip:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:hip:navatala_vector_search_scalar_dequantize_i8_to_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_scalar_dequantize_i8_to_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_compute_quantized_distances_i8[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_dataset", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scale", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_compute_quantized_distances_i8 = {
  1,
  "navatala_vector_search_compute_quantized_distances_i8",
  "hip",
  "navatala_vector_search_compute_quantized_distances_i8",
  "kernel:hip:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:hip:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:hip:navatala_vector_search_compute_quantized_distances_i8",
  7,
  kAbiArgs_hip_navatala_vector_search_compute_quantized_distances_i8
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_encode_p_q_vectors_f32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codewords", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_encode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_encode_p_q_vectors_f32",
  "hip",
  "navatala_vector_search_encode_p_q_vectors_f32",
  "kernel:hip:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:hip:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:hip:navatala_vector_search_encode_p_q_vectors_f32",
  6,
  kAbiArgs_hip_navatala_vector_search_encode_p_q_vectors_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_vector_search_decode_p_q_vectors_f32[] = {
  { "codes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_vector_search_decode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_decode_p_q_vectors_f32",
  "hip",
  "navatala_vector_search_decode_p_q_vectors_f32",
  "kernel:hip:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:hip:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:hip:navatala_vector_search_decode_p_q_vectors_f32",
  5,
  kAbiArgs_hip_navatala_vector_search_decode_p_q_vectors_f32
};

bool tryGetKernelAbiManifest_hip_vector_search(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_init_seeds_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_expand_neighbors_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_check_visited") {
    out = &kAbiManifest_hip_navatala_vector_search_check_visited;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_visited") {
    out = &kAbiManifest_hip_navatala_vector_search_clear_visited;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_extract_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_distances_batch_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_search_layer_greedy_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_select_neighbors_simple_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_select_neighbors_heuristic_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_update_candidate_list_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_extract_layer_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_init_search_state_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out = &kAbiManifest_hip_navatala_vector_search_mark_visited_batch;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_check_visited_batch") {
    out = &kAbiManifest_hip_navatala_vector_search_check_visited_batch;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_merge_multi_query_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_neighbor_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_greedy_search_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_robust_prune_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_insert_vertex_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_neighbors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_update_neighbors_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_random_init_neighbors_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_extract_search_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_batched_greedy_search_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_centroid_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_probes_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_select_probes_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_scan_inverted_list_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_probe_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_merge_probe_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_assign_to_cluster_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_count_cluster_sizes") {
    out = &kAbiManifest_hip_navatala_vector_search_count_cluster_sizes;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_populate_inverted_lists_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_populate_inverted_lists_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out = &kAbiManifest_hip_navatala_vector_search_initialize_random_graph;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_graph_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_nn_descent_join_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_nn_descent_join_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_build_reverse_graph") {
    out = &kAbiManifest_hip_navatala_vector_search_build_reverse_graph;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_prune_graph_r_n_g_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out = &kAbiManifest_hip_navatala_vector_search_compact_pruned_graph;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_vertex_degrees;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out = &kAbiManifest_hip_navatala_vector_search_prune_excess_degree;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_reorder_vertices_by_degree") {
    out = &kAbiManifest_hip_navatala_vector_search_reorder_vertices_by_degree;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out = &kAbiManifest_hip_navatala_vector_search_compact_graph_edges;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out = &kAbiManifest_hip_navatala_vector_search_validate_graph_integrity;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out = &kAbiManifest_hip_navatala_vector_search_prune_excess_degree_sorted;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_accumulate_p_q_codebook_i32") {
    out = &kAbiManifest_hip_navatala_vector_search_accumulate_p_q_codebook_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_normalize_p_q_codebook_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_train_p_q_codebook_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_train_p_q_codebook_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_l2_squared_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_inner_product_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_inner_product_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_l1_distance_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_batch_l2_to_neighbors_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out = &kAbiManifest_hip_navatala_vector_search_l2_squared_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_cosine_similarity_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_linf_distance_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out = &kAbiManifest_hip_navatala_vector_search_hamming_distance_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_correlation_distance_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out = &kAbiManifest_hip_navatala_vector_search_jaccard_distance_u32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_assign_labels_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_accumulate_cluster_sums_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_accumulate_cluster_sums_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_update_centroids_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_clear_cluster_sums_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out = &kAbiManifest_hip_navatala_vector_search_clear_cluster_counts;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_reduce_inertia_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_bitonic_sort_step_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_per_row_top_k_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_merge_sorted_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_rerank_with_exact_distances_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_filter_by_threshold_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_merge_search_results_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_recall_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out = &kAbiManifest_hip_navatala_vector_search_scalar_quantize_f32_to_i8;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_scalar_dequantize_i8_to_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out = &kAbiManifest_hip_navatala_vector_search_compute_quantized_distances_i8;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_encode_p_q_vectors_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out = &kAbiManifest_hip_navatala_vector_search_decode_p_q_vectors_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_vector_search(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_init_seeds_f32";
    std::string_view sv(k_hip_navatala_vector_search_init_seeds_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_expand_neighbors_f32";
    std::string_view sv(k_hip_navatala_vector_search_expand_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_check_visited") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_check_visited";
    std::string_view sv(k_hip_navatala_vector_search_check_visited);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_visited") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_clear_visited";
    std::string_view sv(k_hip_navatala_vector_search_clear_visited);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_extract_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_extract_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_distances_batch_f32";
    std::string_view sv(k_hip_navatala_vector_search_compute_distances_batch_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_search_layer_greedy_f32";
    std::string_view sv(k_hip_navatala_vector_search_search_layer_greedy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_select_neighbors_simple_f32";
    std::string_view sv(k_hip_navatala_vector_search_select_neighbors_simple_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_select_neighbors_heuristic_f32";
    std::string_view sv(k_hip_navatala_vector_search_select_neighbors_heuristic_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_update_candidate_list_f32";
    std::string_view sv(k_hip_navatala_vector_search_update_candidate_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_extract_layer_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_extract_layer_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_init_search_state_f32";
    std::string_view sv(k_hip_navatala_vector_search_init_search_state_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_mark_visited_batch";
    std::string_view sv(k_hip_navatala_vector_search_mark_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_check_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_check_visited_batch";
    std::string_view sv(k_hip_navatala_vector_search_check_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_merge_multi_query_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_merge_multi_query_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_neighbor_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_compute_neighbor_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_greedy_search_f32";
    std::string_view sv(k_hip_navatala_vector_search_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_robust_prune_f32";
    std::string_view sv(k_hip_navatala_vector_search_robust_prune_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_insert_vertex_f32";
    std::string_view sv(k_hip_navatala_vector_search_insert_vertex_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_update_neighbors_f32";
    std::string_view sv(k_hip_navatala_vector_search_update_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_random_init_neighbors_f32";
    std::string_view sv(k_hip_navatala_vector_search_random_init_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_extract_search_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_extract_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_batched_greedy_search_f32";
    std::string_view sv(k_hip_navatala_vector_search_batched_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_centroid_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_compute_centroid_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_select_probes_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_select_probes_f32";
    std::string_view sv(k_hip_navatala_vector_search_select_probes_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_scan_inverted_list_f32";
    std::string_view sv(k_hip_navatala_vector_search_scan_inverted_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_probe_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_merge_probe_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_merge_probe_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_assign_to_cluster_f32";
    std::string_view sv(k_hip_navatala_vector_search_assign_to_cluster_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_count_cluster_sizes") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_count_cluster_sizes";
    std::string_view sv(k_hip_navatala_vector_search_count_cluster_sizes);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_populate_inverted_lists_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_populate_inverted_lists_f32";
    std::string_view sv(k_hip_navatala_vector_search_populate_inverted_lists_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_initialize_random_graph";
    std::string_view sv(k_hip_navatala_vector_search_initialize_random_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_graph_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_compute_graph_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_nn_descent_join_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_nn_descent_join_f32";
    std::string_view sv(k_hip_navatala_vector_search_nn_descent_join_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_build_reverse_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_build_reverse_graph";
    std::string_view sv(k_hip_navatala_vector_search_build_reverse_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_prune_graph_r_n_g_f32";
    std::string_view sv(k_hip_navatala_vector_search_prune_graph_r_n_g_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compact_pruned_graph";
    std::string_view sv(k_hip_navatala_vector_search_compact_pruned_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_vertex_degrees";
    std::string_view sv(k_hip_navatala_vector_search_compute_vertex_degrees);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_prune_excess_degree";
    std::string_view sv(k_hip_navatala_vector_search_prune_excess_degree);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_reorder_vertices_by_degree") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_reorder_vertices_by_degree";
    std::string_view sv(k_hip_navatala_vector_search_reorder_vertices_by_degree);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compact_graph_edges";
    std::string_view sv(k_hip_navatala_vector_search_compact_graph_edges);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_validate_graph_integrity";
    std::string_view sv(k_hip_navatala_vector_search_validate_graph_integrity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_prune_excess_degree_sorted";
    std::string_view sv(k_hip_navatala_vector_search_prune_excess_degree_sorted);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_accumulate_p_q_codebook_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_accumulate_p_q_codebook_i32";
    std::string_view sv(k_hip_navatala_vector_search_accumulate_p_q_codebook_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_normalize_p_q_codebook_f32";
    std::string_view sv(k_hip_navatala_vector_search_normalize_p_q_codebook_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_train_p_q_codebook_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_train_p_q_codebook_f32";
    std::string_view sv(k_hip_navatala_vector_search_train_p_q_codebook_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_l2_squared_f32";
    std::string_view sv(k_hip_navatala_vector_search_l2_squared_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_inner_product_f32";
    std::string_view sv(k_hip_navatala_vector_search_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_l1_distance_f32";
    std::string_view sv(k_hip_navatala_vector_search_l1_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_batch_l2_to_neighbors_f32";
    std::string_view sv(k_hip_navatala_vector_search_batch_l2_to_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_l2_squared_f16";
    std::string_view sv(k_hip_navatala_vector_search_l2_squared_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_cosine_similarity_f32";
    std::string_view sv(k_hip_navatala_vector_search_cosine_similarity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_linf_distance_f32";
    std::string_view sv(k_hip_navatala_vector_search_linf_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_hamming_distance_u32";
    std::string_view sv(k_hip_navatala_vector_search_hamming_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_correlation_distance_f32";
    std::string_view sv(k_hip_navatala_vector_search_correlation_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_jaccard_distance_u32";
    std::string_view sv(k_hip_navatala_vector_search_jaccard_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_assign_labels_f32";
    std::string_view sv(k_hip_navatala_vector_search_assign_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_accumulate_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_accumulate_cluster_sums_f32";
    std::string_view sv(k_hip_navatala_vector_search_accumulate_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_update_centroids_f32";
    std::string_view sv(k_hip_navatala_vector_search_update_centroids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_clear_cluster_sums_f32";
    std::string_view sv(k_hip_navatala_vector_search_clear_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_clear_cluster_counts";
    std::string_view sv(k_hip_navatala_vector_search_clear_cluster_counts);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_reduce_inertia_f32";
    std::string_view sv(k_hip_navatala_vector_search_reduce_inertia_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_bitonic_sort_step_f32";
    std::string_view sv(k_hip_navatala_vector_search_bitonic_sort_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_per_row_top_k_f32";
    std::string_view sv(k_hip_navatala_vector_search_per_row_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_merge_sorted_f32";
    std::string_view sv(k_hip_navatala_vector_search_merge_sorted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_rerank_with_exact_distances_f32";
    std::string_view sv(k_hip_navatala_vector_search_rerank_with_exact_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_filter_by_threshold_f32";
    std::string_view sv(k_hip_navatala_vector_search_filter_by_threshold_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_merge_search_results_f32";
    std::string_view sv(k_hip_navatala_vector_search_merge_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_recall_f32";
    std::string_view sv(k_hip_navatala_vector_search_compute_recall_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_scalar_quantize_f32_to_i8";
    std::string_view sv(k_hip_navatala_vector_search_scalar_quantize_f32_to_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_scalar_dequantize_i8_to_f32";
    std::string_view sv(k_hip_navatala_vector_search_scalar_dequantize_i8_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_compute_quantized_distances_i8";
    std::string_view sv(k_hip_navatala_vector_search_compute_quantized_distances_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_encode_p_q_vectors_f32";
    std::string_view sv(k_hip_navatala_vector_search_encode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_vector_search_decode_p_q_vectors_f32";
    std::string_view sv(k_hip_navatala_vector_search_decode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

