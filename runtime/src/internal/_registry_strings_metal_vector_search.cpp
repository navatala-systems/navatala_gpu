// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_vector_search` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_vector_search_init_seeds_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_init_seeds_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* seed_indices [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* n_seeds [[buffer(4)]], device const uint* dim [[buffer(5)]], device const uint* itopk_size [[buffer(6)]], device uint* candidates [[buffer(7)]], device float* distances [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint ns = n_seeds[0];
  uint d = dim[0];
  uint itopk = itopk_size[0];
  uint total = (nq * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / ns);
    uint seed_local = (gid % ns);
    uint node_idx = seed_indices[seed_local];
    float dist_acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      uint d_idx = ((node_idx * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    uint out_idx = ((query_idx * itopk) + seed_local);
    candidates[out_idx] = node_idx;
    float final_dist = dist_acc;
    distances[out_idx] = final_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_expand_neighbors_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_expand_neighbors_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* top_candidates [[buffer(3)]], device const uint* n_queries [[buffer(4)]], device const uint* search_width [[buffer(5)]], device const uint* graph_degree [[buffer(6)]], device const uint* dim [[buffer(7)]], device uint* new_candidates [[buffer(8)]], device float* new_distances [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint sw = search_width[0];
  uint gd = graph_degree[0];
  uint d = dim[0];
  uint neighbors_per_query = (sw * gd);
  uint total = (nq * neighbors_per_query);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / neighbors_per_query);
    uint local_idx = (gid % neighbors_per_query);
    uint candidate_idx = (local_idx / gd);
    uint neighbor_idx = (local_idx % gd);
    uint tc_idx = ((query_idx * sw) + candidate_idx);
    uint parent_node = top_candidates[tc_idx];
    uint graph_idx = ((parent_node * gd) + neighbor_idx);
    uint neighbor_node = graph[graph_idx];
    float dist_acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_idx * d) + k);
      float q_val = queries[q_idx];
      uint d_idx = ((neighbor_node * d) + k);
      float d_val = dataset[d_idx];
      float diff = (q_val - d_val);
      float diff_sq = (diff * diff);
      float old = dist_acc;
      dist_acc = (old + diff_sq);
    }
    uint out_idx = ((query_idx * neighbors_per_query) + local_idx);
    new_candidates[out_idx] = neighbor_node;
    float final_dist = dist_acc;
    new_distances[out_idx] = final_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_check_visited = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_check_visited(device const uint* candidates [[buffer(0)]], device atomic_uint* visited [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* hash_size [[buffer(3)]], device uint* is_new [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint num = n[0];
  uint hs = hash_size[0];
  bool inBounds = (gid < num);
  if (inBounds) {
    uint node = candidates[gid];
    uint hash_idx = (node % hs);
    uint old_val = gpu_atomic_cas_uint((device atomic_uint*)(&(visited[hash_idx])), (uint)(0u), (uint)((node + 1u)));
    bool was_empty = (old_val == 0u);
    uint result = ((was_empty) ? (1u) : (0u));
    is_new[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_clear_visited = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_clear_visited(device const uint* hash_size [[buffer(0)]], device uint* visited [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint hs = hash_size[0];
  bool inBounds = (gid < hs);
  if (inBounds) {
    visited[gid] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_extract_results_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_extract_results_f32(device const uint* sorted_candidates [[buffer(0)]], device const float* sorted_distances [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* itopk_size [[buffer(4)]], device uint* result_indices [[buffer(5)]], device float* result_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint kVal = k[0];
  uint itopk = itopk_size[0];
  uint total = (nq * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint query_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint src_idx = ((query_idx * itopk) + k_idx);
    uint idx_val = sorted_candidates[src_idx];
    float dist_val = sorted_distances[src_idx];
    result_indices[gid] = idx_val;
    result_distances[gid] = dist_val;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_distances_batch_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_distances_batch_f32(device const float* query [[buffer(0)]], device const float* candidates [[buffer(1)]], device const uint* candidate_ids [[buffer(2)]], device const uint* n_candidates [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n_cand = n_candidates[0];
  uint d = dim[0];
  if (gid < n_cand) {
    uint cand_id = candidate_ids[gid];
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      uint c_idx = ((cand_id * d) + k);
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
const char* k_metal_navatala_vector_search_search_layer_greedy_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_search_layer_greedy_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* graph_degrees [[buffer(3)]], device const uint* entry_points [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* dim [[buffer(6)]], device const uint* ef [[buffer(7)]], device const uint* max_degree [[buffer(8)]], device uint* candidates [[buffer(9)]], device float* candidate_distances [[buffer(10)]], device uint* n_found [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint d = dim[0];
  if (query_id < nq) {
    uint entry = entry_points[query_id];
    float best_dist = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_id * d) + k);
      uint e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_bd = best_dist;
      best_dist = (old_bd + sq);
    }
    uint ef_val = ef[0];
    uint cand_idx = (query_id * ef_val);
    candidates[cand_idx] = entry;
    float init_dist = best_dist;
    candidate_distances[cand_idx] = init_dist;
    n_found[query_id] = 1u;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_select_neighbors_simple_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_select_neighbors_simple_f32(device const uint* candidates [[buffer(0)]], device const float* distances [[buffer(1)]], device const uint* n_candidates [[buffer(2)]], device const uint* m [[buffer(3)]], device uint* selected [[buffer(4)]], device uint* selected_count [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint n_cand = n_candidates[0];
  uint m_val = m[0];
  if (tid == 0u) {
    uint limit = (((n_cand < m_val)) ? (n_cand) : (m_val));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    selected_count[0u] = limit;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_select_neighbors_heuristic_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_select_neighbors_heuristic_f32(device const float* query [[buffer(0)]], device const uint* candidates [[buffer(1)]], device const float* distances [[buffer(2)]], device const float* dataset [[buffer(3)]], device const uint* n_candidates [[buffer(4)]], device const uint* dim [[buffer(5)]], device const uint* m [[buffer(6)]], device uint* selected [[buffer(7)]], device uint* n_selected [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint n_cand = n_candidates[0];
  uint m_val = m[0];
  threadgroup uint selected_ids[64];
  threadgroup uint n_sel[1];
  if (tid == 0u) {
    n_sel[0u] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < n_cand) {
    uint cand_id = candidates[tid];
    selected[tid] = cand_id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint final_count = (((n_cand < m_val)) ? (n_cand) : (m_val));
    n_selected[0u] = final_count;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_update_candidate_list_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_update_candidate_list_f32(device uint* candidates [[buffer(0)]], device float* distances [[buffer(1)]], device const uint* new_candidates [[buffer(2)]], device const float* new_distances [[buffer(3)]], device const uint* n_current [[buffer(4)]], device const uint* n_new [[buffer(5)]], device const uint* ef [[buffer(6)]], device uint* n_merged [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint n_curr = n_current[0];
  uint n_new_val = n_new[0];
  uint ef_val = ef[0];
  threadgroup uint merge_ids[512];
  threadgroup float merge_dists[512];
  if (tid < n_curr) {
    uint cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < n_new_val) {
    uint offset = (n_curr + tid);
    uint nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint total = (n_curr + n_new_val);
    uint final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[0u] = final_n;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_extract_layer_results_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_extract_layer_results_f32(device const uint* candidates [[buffer(0)]], device const float* distances [[buffer(1)]], device const uint* n_found [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* k [[buffer(4)]], device uint* result_ids [[buffer(5)]], device float* result_dists [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint query_id = (gid / k_val);
  uint neighbor_idx = (gid % k_val);
  if (query_id < nq) {
    uint found = n_found[query_id];
    uint src_idx = ((query_id * k_val) + neighbor_idx);
    if (neighbor_idx < found) {
      uint cand_id = candidates[src_idx];
      float cand_dist = distances[src_idx];
      result_ids[src_idx] = cand_id;
      result_dists[src_idx] = cand_dist;
    } else {
      result_ids[src_idx] = 4294967295u;
      result_dists[src_idx] = as_type<float>(0x7e967699u);
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_init_search_state_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_init_search_state_f32(device const uint* global_entry [[buffer(0)]], device const uint* n_queries [[buffer(1)]], device const uint* visited_size [[buffer(2)]], device uint* entry_points [[buffer(3)]], device uint* visited [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint entry = global_entry[0];
  uint nq = n_queries[0];
  uint vs = visited_size[0];
  if (gid < nq) {
    entry_points[gid] = entry;
  }
  uint total_visited = (nq * vs);
  if (gid < total_visited) {
    uint q_idx = (gid / vs);
    uint v_idx = (gid % vs);
    uint flat_idx = ((q_idx * vs) + v_idx);
    visited[flat_idx] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_mark_visited_batch = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_mark_visited_batch(device const uint* node_ids [[buffer(0)]], device const uint* query_ids [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* visited_stride [[buffer(3)]], device uint* visited [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if (tid < nn) {
    uint node = node_ids[tid];
    uint query = query_ids[tid];
    uint word_idx = (node / 32u);
    uint bit_idx = (node % 32u);
    uint bit_mask = (1u << bit_idx);
    uint flat_idx = ((query * stride) + word_idx);
    atomic_fetch_or_explicit((device atomic_uint*)(&visited[flat_idx]), bit_mask, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_check_visited_batch = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_check_visited_batch(device const uint* node_ids [[buffer(0)]], device const uint* query_ids [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* visited_stride [[buffer(3)]], device const uint* visited [[buffer(4)]], device uint* is_visited [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if (tid < nn) {
    uint node = node_ids[tid];
    uint query = query_ids[tid];
    uint word_idx = (node / 32u);
    uint bit_idx = (node % 32u);
    uint bit_mask = (1u << bit_idx);
    uint flat_idx = ((query * stride) + word_idx);
    uint word = visited[flat_idx];
    uint is_set = (word & bit_mask);
    uint result = (((is_set != 0u)) ? (1u) : (0u));
    is_visited[tid] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_merge_multi_query_results_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_merge_multi_query_results_f32(device const uint* layer_results [[buffer(0)]], device const float* layer_distances [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* n_layers [[buffer(4)]], device uint* final_ids [[buffer(5)]], device float* final_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if (query_id < nq) {
    uint flat_idx = ((query_id * k_val) + k_idx);
    uint result_id = layer_results[flat_idx];
    float result_dist = layer_distances[flat_idx];
    final_ids[flat_idx] = result_id;
    final_distances[flat_idx] = result_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_neighbor_distances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_neighbor_distances_f32(device const float* query [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* neighbor_ids [[buffer(2)]], device const uint* n_neighbors [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nn = n_neighbors[0];
  uint d = dim[0];
  if (tid < nn) {
    uint neighbor_id = neighbor_ids[tid];
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      uint n_idx = ((neighbor_id * d) + k);
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
const char* k_metal_navatala_vector_search_greedy_search_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_greedy_search_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* graph_degrees [[buffer(3)]], device const uint* entry_point [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* dim [[buffer(6)]], device const uint* max_degree [[buffer(7)]], device const uint* search_list_size [[buffer(8)]], device uint* candidates [[buffer(9)]], device float* candidate_dists [[buffer(10)]], device uint* n_candidates [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint ls = search_list_size[0];
  if (query_id < nq) {
    float entry_dist = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((query_id * d) + k);
      uint e_idx = ((entry * d) + k);
      float q_val = queries[q_idx];
      float e_val = dataset[e_idx];
      float diff = (q_val - e_val);
      float sq = (diff * diff);
      float old_ed = entry_dist;
      entry_dist = (old_ed + sq);
    }
    uint cand_base = (query_id * ls);
    candidates[cand_base] = entry;
    float ed = entry_dist;
    candidate_dists[cand_base] = ed;
    n_candidates[query_id] = 1u;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_robust_prune_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_robust_prune_f32(device const uint* vertex [[buffer(0)]], device const uint* candidates [[buffer(1)]], device const float* distances [[buffer(2)]], device const float* dataset [[buffer(3)]], device const uint* n_candidates [[buffer(4)]], device const uint* dim [[buffer(5)]], device const uint* max_degree [[buffer(6)]], device const uint* alpha [[buffer(7)]], device uint* selected [[buffer(8)]], device uint* n_selected [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  if (tid == 0u) {
    uint limit = (((nc < md)) ? (nc) : (md));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    n_selected[0u] = limit;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_insert_vertex_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_insert_vertex_f32(device const uint* vertex_id [[buffer(0)]], device const uint* candidates [[buffer(1)]], device const uint* n_candidates [[buffer(2)]], device const uint* max_degree [[buffer(3)]], device uint* graph [[buffer(4)]], device uint* graph_degrees [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint vid = vertex_id[0];
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  uint n_neighbors = (((nc < md)) ? (nc) : (md));
  if (tid < n_neighbors) {
    uint neighbor = candidates[tid];
    uint graph_idx = ((vid * md) + tid);
    graph[graph_idx] = neighbor;
  }
  if (tid == 0u) {
    graph_degrees[vid] = n_neighbors;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_random_init_neighbors_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_random_init_neighbors_f32(device const uint* n_vectors [[buffer(0)]], device const uint* max_degree [[buffer(1)]], device const uint* init_degree [[buffer(2)]], device const uint* seed [[buffer(3)]], device uint* graph [[buffer(4)]], device uint* graph_degrees [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint md = max_degree[0];
  uint id = init_degree[0];
  uint s = seed[0];
  if (vid < nv) {
    uint rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      uint old_rng = rng;
      uint temp1 = (old_rng * 1103515245u);
      uint temp2 = (temp1 + 12345u);
      uint new_rng = (temp2 & 2147483647u);
      rng = new_rng;
      uint neighbor_raw = (new_rng % nv);
      uint neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + 1u) % nv)) : (neighbor_raw));
      uint graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_extract_search_results_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_extract_search_results_f32(device const uint* candidates [[buffer(0)]], device const float* candidate_dists [[buffer(1)]], device const uint* n_candidates [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* search_list_size [[buffer(5)]], device uint* result_ids [[buffer(6)]], device float* result_dists [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint ls = search_list_size[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if (query_id < nq) {
    uint nc = n_candidates[query_id];
    uint src_idx = ((query_id * ls) + k_idx);
    uint dst_idx = ((query_id * k_val) + k_idx);
    if (k_idx < nc) {
      uint cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = 4294967295u;
      result_dists[dst_idx] = as_type<float>(0x7e967699u);
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_batched_greedy_search_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_batched_greedy_search_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* graph [[buffer(2)]], device const uint* graph_degrees [[buffer(3)]], device const uint* entry_point [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* n_vectors [[buffer(6)]], device const uint* dim [[buffer(7)]], device const uint* max_degree [[buffer(8)]], device const uint* k [[buffer(9)]], device uint* result_ids [[buffer(10)]], device float* result_dists [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__tgid.x)));
  uint tid = ((uint)(int(__tid.x)));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint k_val = k[0];
  threadgroup float partial_sums[64];
  if (query_id < nq) {
    if (tid == 0u) {
      uint base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = as_type<float>(0x00000000u);
    }
    if (tid < k_val) {
      if (tid != 0u) {
        uint out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = 4294967295u;
        result_dists[out_idx] = as_type<float>(0x7e967699u);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_centroid_distances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_centroid_distances_f32(device const float* queries [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* n_lists [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* centroid_distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  uint total_pairs = (nq * nl);
  bool inBounds = (gid < total_pairs);
  if (inBounds) {
    uint query_idx = (gid / nl);
    uint list_idx = (gid % nl);
    uint query_base = (query_idx * d);
    uint centroid_base = (list_idx * d);
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_offset = (query_base + k);
      uint c_offset = (centroid_base + k);
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
const char* k_metal_navatala_vector_search_scan_inverted_list_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_scan_inverted_list_f32(device const float* queries [[buffer(0)]], device const float* list_data [[buffer(1)]], device const uint* list_indices [[buffer(2)]], device const uint* list_offsets [[buffer(3)]], device const uint* probe_indices [[buffer(4)]], device const uint* n_queries [[buffer(5)]], device const uint* n_probes [[buffer(6)]], device const uint* dim [[buffer(7)]], device const uint* max_list_size [[buffer(8)]], device float* scan_distances [[buffer(9)]], device uint* scan_indices [[buffer(10)]], device uint* scan_counts [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  uint wg_size = 256u;
  uint nq = n_queries[0];
  uint np = n_probes[0];
  uint d = dim[0];
  uint mls = max_list_size[0];
  uint total_pairs = (nq * np);
  bool valid_wg = (wgid < total_pairs);
  if (valid_wg) {
    uint query_idx = (wgid / np);
    uint probe_idx = (wgid % np);
    uint probe_offset = ((query_idx * np) + probe_idx);
    uint list_idx = probe_indices[probe_offset];
    uint list_start = list_offsets[list_idx];
    uint list_idx_plus1 = (list_idx + 1u);
    uint list_end = list_offsets[list_idx_plus1];
    uint list_size = (list_end - list_start);
    uint query_base = (query_idx * d);
    uint out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    uint iters = ((list_size / wg_size) + 1u);
    for (int iter = 0; iter < (int)(iters); ++iter) {
      uint local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        uint global_vec_idx = (list_start + local_vec_idx);
        uint vec_base = (global_vec_idx * d);
        float dist_acc = as_type<float>(0x00000000u);
        for (int k = 0; k < (int)(d); ++k) {
          uint q_off = (query_base + k);
          uint v_off = (vec_base + k);
          float q_val = queries[q_off];
          float v_val = list_data[v_off];
          float diff = (q_val - v_val);
          float sq = (diff * diff);
          float old_d = dist_acc;
          dist_acc = (old_d + sq);
        }
        uint orig_idx = list_indices[global_vec_idx];
        uint out_offset = (out_base + local_vec_idx);
        float final_d = dist_acc;
        scan_distances[out_offset] = final_d;
        scan_indices[out_offset] = orig_idx;
      }
    }
    bool is_t0 = (tid == 0u);
    if (is_t0) {
      uint count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_assign_to_cluster_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_assign_to_cluster_f32(device const float* vectors [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* n_lists [[buffer(3)]], device const uint* dim [[buffer(4)]], device uint* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint vec_base = (gid * d);
    float best_dist = as_type<float>(0x7f7fc99eu);
    uint best_label = 0u;
    for (int c = 0; c < (int)(nl); ++c) {
      uint cent_base = (c * d);
      float dist_acc = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        uint v_off = (vec_base + k);
        uint c_off = (cent_base + k);
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
    uint final_label = best_label;
    labels[gid] = final_label;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_count_cluster_sizes = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_count_cluster_sizes(device const uint* labels [[buffer(0)]], device const uint* n_vectors [[buffer(1)]], device const uint* n_lists [[buffer(2)]], device atomic_uint* cluster_counts [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint label = labels[gid];
    uint _old_count = atomic_fetch_add_explicit(((device atomic_uint*)(&(cluster_counts[label]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_populate_inverted_lists_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_populate_inverted_lists_f32(device const float* vectors [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* list_offsets [[buffer(2)]], device const uint* n_vectors [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* list_data [[buffer(5)]], device uint* list_indices [[buffer(6)]], device atomic_uint* write_positions [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint label = labels[gid];
    uint list_base = list_offsets[label];
    uint local_pos = atomic_fetch_add_explicit(((device atomic_uint*)(&(write_positions[label]))), 1u, memory_order_relaxed);
    uint global_pos = (list_base + local_pos);
    uint src_base = (gid * d);
    uint dst_base = (global_pos * d);
    for (int k = 0; k < (int)(d); ++k) {
      uint src_off = (src_base + k);
      uint dst_off = (dst_base + k);
      float val = vectors[src_off];
      list_data[dst_off] = val;
    }
    list_indices[global_pos] = gid;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_initialize_random_graph = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_initialize_random_graph(device const uint* n_nodes [[buffer(0)]], device const uint* graph_degree [[buffer(1)]], device const uint* seed [[buffer(2)]], device uint* graph [[buffer(3)]], device float* graph_distances [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint s = seed[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint slot = (gid % deg);
    uint state0 = ((gid * 1103515245u) + (s + 12345u));
    uint rand_raw = (state0 % nn);
    bool is_self = (rand_raw == node_idx);
    uint alt = ((node_idx + 1u) % nn);
    uint neighbor = ((is_self) ? (alt) : (rand_raw));
    graph[gid] = neighbor;
    graph_distances[gid] = as_type<float>(0x7f7fc99eu);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_graph_distances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_graph_distances_f32(device const float* dataset [[buffer(0)]], device const uint* graph [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* graph_degree [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* graph_distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint d = dim[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint neighbor = graph[gid];
    uint node_base = (node_idx * d);
    uint neighbor_base = (neighbor * d);
    float dist_acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint n_off = (node_base + k);
      uint nb_off = (neighbor_base + k);
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
const char* k_metal_navatala_vector_search_nn_descent_join_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_nn_descent_join_f32(device const float* dataset [[buffer(0)]], device const uint* graph [[buffer(1)]], device const float* graph_distances [[buffer(2)]], device const uint* n_nodes [[buffer(3)]], device const uint* graph_degree [[buffer(4)]], device const uint* dim [[buffer(5)]], device uint* new_graph [[buffer(6)]], device float* new_distances [[buffer(7)]], device atomic_uint* update_count [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint node_idx = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint d = dim[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    uint node_base = (node_idx * d);
    uint graph_base = (node_idx * deg);
    float worst_dist = as_type<float>(0x00000000u);
    uint worst_slot = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      uint slot_idx = (graph_base + i);
      float slot_dist = graph_distances[slot_idx];
      float cur_worst = worst_dist;
      bool is_worse = (slot_dist > cur_worst);
      if (is_worse) {
        worst_dist = slot_dist;
        worst_slot = i;
      }
    }
    for (int ni = 0; ni < (int)(deg); ++ni) {
      uint neighbor_idx_slot = (graph_base + ni);
      uint neighbor = graph[neighbor_idx_slot];
      uint neighbor_graph_base = (neighbor * deg);
      for (int nni = 0; nni < (int)(deg); ++nni) {
        uint nn_slot = (neighbor_graph_base + nni);
        uint candidate = graph[nn_slot];
        bool is_self = (candidate == node_idx);
        if ((candidate != node_idx) && true) {
          uint cand_base = (candidate * d);
          float cand_dist_acc = as_type<float>(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            uint n_off = (node_base + k);
            uint c_off = (cand_base + k);
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
            uint update_slot = worst_slot;
            uint update_idx = (graph_base + update_slot);
            new_graph[update_idx] = candidate;
            new_distances[update_idx] = cand_dist;
            worst_dist = cand_dist;
            uint _upd = atomic_fetch_add_explicit(((device atomic_uint*)(&(update_count[0u]))), 1u, memory_order_relaxed);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_build_reverse_graph = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_build_reverse_graph(device const uint* graph [[buffer(0)]], device const uint* n_nodes [[buffer(1)]], device const uint* graph_degree [[buffer(2)]], device uint* reverse_graph [[buffer(3)]], device atomic_uint* reverse_counts [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint source = (gid / deg);
    uint target = graph[gid];
    uint rev_slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(reverse_counts[target]))), 1u, memory_order_relaxed);
    bool has_space = (rev_slot < deg);
    if (has_space) {
      uint rev_idx = ((target * deg) + rev_slot);
      reverse_graph[rev_idx] = source;
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_prune_graph_r_n_g_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_prune_graph_r_n_g_f32(device const float* dataset [[buffer(0)]], device const uint* graph [[buffer(1)]], device const float* graph_distances [[buffer(2)]], device const uint* n_nodes [[buffer(3)]], device const uint* graph_degree [[buffer(4)]], device const uint* dim [[buffer(5)]], device uint* pruned_graph [[buffer(6)]], device float* pruned_distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint d = dim[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint slot = (gid % deg);
    uint graph_base = (node_idx * deg);
    uint target = graph[gid];
    float edge_dist = graph_distances[gid];
    uint target_base = (target * d);
    bool should_prune = false;
    for (int wi = 0; wi < (int)(deg); ++wi) {
      bool is_self_slot = (wi == slot);
      if (wi != slot) {
        uint w_slot = (graph_base + wi);
        uint w = graph[w_slot];
        float d_uw = graph_distances[w_slot];
        bool uw_closer = (d_uw < edge_dist);
        if (uw_closer) {
          uint w_base = (w * d);
          float d_wv_acc = as_type<float>(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            uint w_off = (w_base + k);
            uint t_off = (target_base + k);
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
    uint invalid_marker = 4294967295u;
    uint out_idx = ((prune) ? (invalid_marker) : (target));
    float out_dist = ((prune) ? (as_type<float>(0x7f7fc99eu)) : (edge_dist));
    pruned_graph[gid] = out_idx;
    pruned_distances[gid] = out_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compact_pruned_graph = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compact_pruned_graph(device const uint* pruned_graph [[buffer(0)]], device const float* pruned_distances [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* graph_degree [[buffer(3)]], device uint* compact_graph [[buffer(4)]], device float* compact_distances [[buffer(5)]], device uint* degree_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint node_idx = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    uint base = (node_idx * deg);
    uint write_pos = 0u;
    for (int i = 0; i < (int)(deg); ++i) {
      uint read_idx = (base + i);
      uint neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != 4294967295u);
      if (is_valid) {
        uint wp = write_pos;
        uint write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + 1u);
      }
    }
    uint final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_vertex_degrees = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_vertex_degrees(device const uint* graph [[buffer(0)]], device const uint* n_vertices [[buffer(1)]], device const uint* max_degree [[buffer(2)]], device const uint* invalid_id [[buffer(3)]], device uint* degrees [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  uint inv = invalid_id[0];
  if (vid < nv) {
    uint count = 0u;
    for (int i = 0; i < (int)(md); ++i) {
      uint idx = ((vid * md) + i);
      uint neighbor = graph[idx];
      if (neighbor != inv) {
        uint old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    uint final_count = count;
    degrees[vid] = final_count;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_prune_excess_degree = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_prune_excess_degree(device uint* graph [[buffer(0)]], device const float* neighbor_distances [[buffer(1)]], device uint* degrees [[buffer(2)]], device const uint* n_vertices [[buffer(3)]], device const uint* current_max_degree [[buffer(4)]], device const uint* target_max_degree [[buffer(5)]], device const uint* invalid_id [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint cmd = current_max_degree[0];
  uint tmd = target_max_degree[0];
  uint inv = invalid_id[0];
  if (vid < nv) {
    uint deg = degrees[vid];
    if (deg > tmd) {
      for (int i = 0; i < (int)(cmd); ++i) {
        if (i >= tmd) {
          uint idx = ((vid * cmd) + i);
          graph[idx] = inv;
        }
      }
      degrees[vid] = tmd;
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_reorder_vertices_by_degree = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_reorder_vertices_by_degree(device const uint* degrees [[buffer(0)]], device atomic_uint* prefix_sums [[buffer(1)]], device const uint* n_vertices [[buffer(2)]], device const uint* n_bins [[buffer(3)]], device uint* old_to_new [[buffer(4)]], device uint* new_to_old [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint nb = n_bins[0];
  if (vid < nv) {
    uint deg = degrees[vid];
    uint bin = (((deg >= nb)) ? ((nb - 1u)) : (deg));
    uint new_id = atomic_fetch_add_explicit(((device atomic_uint*)(&(prefix_sums[bin]))), 1u, memory_order_relaxed);
    old_to_new[vid] = new_id;
    new_to_old[new_id] = vid;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compact_graph_edges = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compact_graph_edges(device uint* graph [[buffer(0)]], device const uint* n_vertices [[buffer(1)]], device const uint* max_degree [[buffer(2)]], device const uint* invalid_id [[buffer(3)]], device uint* degrees [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  uint inv = invalid_id[0];
  if (vid < nv) {
    uint write_pos = 0u;
    for (int i = 0; i < (int)(md); ++i) {
      uint read_idx = ((vid * md) + i);
      uint neighbor = graph[read_idx];
      if (neighbor != inv) {
        uint wp = write_pos;
        uint write_idx = ((vid * md) + wp);
        graph[write_idx] = neighbor;
        write_pos = (wp + 1u);
      }
    }
    uint final_wp = write_pos;
    for (int j = 0; j < (int)(md); ++j) {
      if (j >= final_wp) {
        uint fill_idx = ((vid * md) + j);
        graph[fill_idx] = inv;
      }
    }
    degrees[vid] = final_wp;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_validate_graph_integrity = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_validate_graph_integrity(device const uint* graph [[buffer(0)]], device const uint* degrees [[buffer(1)]], device const uint* n_vertices [[buffer(2)]], device const uint* max_degree [[buffer(3)]], device uint* error_flags [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  if (vid < nv) {
    uint errors = 0u;
    uint deg = degrees[vid];
    for (int i = 0; i < (int)(deg); ++i) {
      uint idx = ((vid * md) + i);
      uint neighbor = graph[idx];
      if (neighbor == vid) {
        uint old_err = errors;
        errors = (old_err | 1u);
      }
      if (neighbor >= nv) {
        uint old_err2 = errors;
        errors = (old_err2 | 2u);
      }
    }
    uint final_errors = errors;
    error_flags[vid] = final_errors;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_prune_excess_degree_sorted = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_prune_excess_degree_sorted(device uint* graph [[buffer(0)]], device const float* neighbor_distances [[buffer(1)]], device uint* degrees [[buffer(2)]], device const uint* n_vertices [[buffer(3)]], device const uint* current_max_degree [[buffer(4)]], device const uint* target_max_degree [[buffer(5)]], device const uint* invalid_id [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__tgid.x)));
  uint tid = ((uint)(int(__tid.x)));
  uint nv = n_vertices[0];
  uint cmd = current_max_degree[0];
  uint tmd = target_max_degree[0];
  uint inv = invalid_id[0];
  threadgroup uint shared_ids[128];
  threadgroup float shared_dists[128];
  if (vid < nv) {
    uint deg = degrees[vid];
    if (deg > tmd) {
      if (tid < deg) {
        uint read_idx = ((vid * cmd) + tid);
        uint neighbor_id = graph[read_idx];
        float neighbor_dist = neighbor_distances[read_idx];
        shared_ids[tid] = neighbor_id;
        shared_dists[tid] = neighbor_dist;
      } else {
        if (tid < 128u) {
          shared_ids[tid] = inv;
          shared_dists[tid] = as_type<float>(0x7e967699u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int stage = 0; stage < (int)(7u); ++stage) {
        for (int substage = 0; substage < (int)((stage + 1u)); ++substage) {
          uint half_net = (1u << (stage - substage));
          uint partner = (tid ^ half_net);
          uint direction_bit = ((tid >> (stage + 1u)) & 1u);
          if (partner < 128u) {
            float my_dist = shared_dists[tid];
            float partner_d = shared_dists[partner];
            uint should_swap = (((tid < partner)) ? ((((direction_bit == 0u)) ? ((((my_dist > partner_d)) ? (1u) : (0u))) : ((((my_dist < partner_d)) ? (1u) : (0u))))) : (0u));
            if (should_swap == 1u) {
              uint my_id = shared_ids[tid];
              uint partner_id = shared_ids[partner];
              shared_ids[tid] = partner_id;
              shared_ids[partner] = my_id;
              shared_dists[tid] = partner_d;
              shared_dists[partner] = my_dist;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
      if (tid < cmd) {
        uint write_idx = ((vid * cmd) + tid);
        if (tid < tmd) {
          uint sorted_id = shared_ids[tid];
          graph[write_idx] = sorted_id;
        } else {
          graph[write_idx] = inv;
        }
      }
      if (tid == 0u) {
        degrees[vid] = tmd;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_accumulate_p_q_codebook_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_accumulate_p_q_codebook_i32(device const float* subvectors [[buffer(0)]], device const uint* assignments [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* sub_dim [[buffer(3)]], device atomic_int* codebook_sums_i32 [[buffer(4)]], device atomic_uint* codebook_counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  float scale = as_type<float>(0x47800000u);
  if (vid < nv) {
    uint cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      uint src_idx = ((vid * sd) + k);
      uint dst_idx = ((cw * sd) + k);
      float val_f32 = subvectors[src_idx];
      float scaled_f32 = (val_f32 * scale);
      int scaled_i32 = ((int)(scaled_f32));
      int _old = atomic_fetch_add_explicit(((device atomic_int*)(&(codebook_sums_i32[dst_idx]))), scaled_i32, memory_order_relaxed);
    }
    uint _cnt = atomic_fetch_add_explicit(((device atomic_uint*)(&(codebook_counts[cw]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_normalize_p_q_codebook_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_normalize_p_q_codebook_f32(device const int* codebook_sums_i32 [[buffer(0)]], device const uint* codebook_counts [[buffer(1)]], device const uint* n_codewords [[buffer(2)]], device const uint* sub_dim [[buffer(3)]], device float* codebook [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  uint total = (nc * sd);
  float scale = as_type<float>(0x47800000u);
  if (tid < total) {
    uint cw = (tid / sd);
    uint count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > as_type<float>(0x00000000u))) ? ((sum_f32 / count_f32)) : (as_type<float>(0x00000000u)));
    codebook[tid] = _centroid;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_train_p_q_codebook_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_train_p_q_codebook_f32(device const float* subvectors [[buffer(0)]], device const uint* assignments [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* n_codewords [[buffer(3)]], device const uint* sub_dim [[buffer(4)]], device float* codebook_sums [[buffer(5)]], device atomic_uint* codebook_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  if (vid < nv) {
    uint cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      uint src_idx = ((vid * sd) + k);
      uint dst_idx = ((cw * sd) + k);
      float val = subvectors[src_idx];
      float old = codebook_sums[dst_idx];
      float new_val = (old + val);
      codebook_sums[dst_idx] = new_val;
    }
    uint _cnt = atomic_fetch_add_explicit(((device atomic_uint*)(&(codebook_counts[cw]))), 1u, memory_order_relaxed);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_l2_squared_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_l2_squared_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    uint dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_inner_product_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_inner_product_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float product = (x_val * y_val);
      float old_acc = acc;
      acc = (old_acc + product);
    }
    uint dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_l1_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_l1_distance_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float neg_diff = (as_type<float>(0x00000000u) - diff);
      bool is_neg = (diff < as_type<float>(0x00000000u));
      float abs_diff = ((is_neg) ? (neg_diff) : (diff));
      float old_acc = acc;
      acc = (old_acc + abs_diff);
    }
    uint dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_batch_l2_to_neighbors_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_batch_l2_to_neighbors_f32(device const float* query [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* indices [[buffer(2)]], device const uint* n_neighbors [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* distances [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = n_neighbors[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint vec_idx = indices[gid];
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      uint d_idx = ((vec_idx * d) + k);
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
const char* k_metal_navatala_vector_search_l2_squared_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_l2_squared_f16(device const half* x [[buffer(0)]], device const half* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      half x_f16 = x[x_idx];
      half y_f16 = y[y_idx];
      float x_val = ((float)(x_f16));
      float y_val = ((float)(y_f16));
      float diff = (x_val - y_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    uint dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_cosine_similarity_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_cosine_similarity_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* sim [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float dot = as_type<float>(0x00000000u);
    float norm_x = as_type<float>(0x00000000u);
    float norm_y = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
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
    float eps = as_type<float>(0x322bcc77u);
    float safe_denom = (denom + eps);
    float cosine = (final_dot / safe_denom);
    uint out_idx = ((gid_x * ny) + gid_y);
    sim[out_idx] = cosine;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_linf_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_linf_distance_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float max_diff = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
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
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_hamming_distance_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_hamming_distance_u32(device const uint* x [[buffer(0)]], device const uint* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim_words [[buffer(4)]], device uint* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint xor_val = (x_val ^ y_val);
      uint v0 = xor_val;
      uint m1 = 1431655765u;
      uint m2 = 858993459u;
      uint m4 = 252645135u;
      uint v1 = (v0 - ((v0 >> 1u) & m1));
      uint v2 = ((v1 & m2) + ((v1 >> 2u) & m2));
      uint v3 = ((v2 + (v2 >> 4u)) & m4);
      uint v4 = (v3 + (v3 >> 8u));
      uint v5 = (v4 + (v4 >> 16u));
      uint popcount = (v5 & 63u);
      uint old_count = count;
      count = (old_count + popcount);
    }
    uint result = count;
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_correlation_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_correlation_distance_f32(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float sum_x = as_type<float>(0x00000000u);
    float sum_y = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
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
    float cov = as_type<float>(0x00000000u);
    float var_x = as_type<float>(0x00000000u);
    float var_y = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx2 = ((gid_x * d) + k);
      uint y_idx2 = ((gid_y * d) + k);
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
    float eps = as_type<float>(0x322bcc77u);
    float safe_denom = (sqrt_var + eps);
    float correlation = (final_cov / safe_denom);
    float corr_dist = (as_type<float>(0x3f800000u) - correlation);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = corr_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_jaccard_distance_u32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_jaccard_distance_u32(device const uint* x [[buffer(0)]], device const uint* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim_words [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint intersection = 0u;
    uint union_count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint inter = (x_val & y_val);
      uint uni = (x_val | y_val);
      uint i0 = inter;
      uint m1 = 1431655765u;
      uint m2 = 858993459u;
      uint m4 = 252645135u;
      uint i1 = (i0 - ((i0 >> 1u) & m1));
      uint i2 = ((i1 & m2) + ((i1 >> 2u) & m2));
      uint i3 = ((i2 + (i2 >> 4u)) & m4);
      uint i4 = (i3 + (i3 >> 8u));
      uint i5 = (i4 + (i4 >> 16u));
      uint pop_inter = (i5 & 63u);
      uint u0 = uni;
      uint u1 = (u0 - ((u0 >> 1u) & m1));
      uint u2 = ((u1 & m2) + ((u1 >> 2u) & m2));
      uint u3 = ((u2 + (u2 >> 4u)) & m4);
      uint u4 = (u3 + (u3 >> 8u));
      uint u5 = (u4 + (u4 >> 16u));
      uint pop_union = (u5 & 63u);
      uint old_inter = intersection;
      uint old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    uint final_inter = intersection;
    uint final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = as_type<float>(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (as_type<float>(0x3f800000u) - similarity);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_assign_labels_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_assign_labels_f32(device const float* data [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_points [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* dim [[buffer(4)]], device uint* labels [[buffer(5)]], device float* min_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = n_points[0];
  uint kVal = k[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float best_dist = as_type<float>(0x7e967699u);
    uint best_label = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      float dist_acc = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(d); ++j) {
        uint data_idx = ((gid * d) + j);
        float data_val = data[data_idx];
        uint cent_idx = ((c * d) + j);
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
    uint final_label = best_label;
    float final_dist = best_dist;
    labels[gid] = final_label;
    min_distances[gid] = final_dist;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_accumulate_cluster_sums_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_vector_search_accumulate_cluster_sums_f32(device const float* data [[buffer(0)]], device const uint* labels [[buffer(1)]], device const uint* n_points [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* dim [[buffer(4)]], device float* cluster_sums [[buffer(5)]], device atomic_uint* cluster_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = n_points[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint label = labels[gid];
    uint _old_count = atomic_fetch_add_explicit(((device atomic_uint*)(&(cluster_counts[label]))), 1u, memory_order_relaxed);
    for (int j = 0; j < (int)(d); ++j) {
      uint data_idx = ((gid * d) + j);
      float data_val = data[data_idx];
      uint sum_idx = ((label * d) + j);
      atomic_fetch_add_explicit((device atomic_float*)(&cluster_sums[sum_idx]), data_val, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_vector_search_update_centroids_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_update_centroids_f32(device const float* cluster_sums [[buffer(0)]], device const uint* cluster_counts [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* dim [[buffer(3)]], device float* centroids [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint d = dim[0];
  uint total = (kVal * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint cluster_idx = (gid / d);
    uint dim_idx = (gid % d);
    uint count = cluster_counts[cluster_idx];
    float sum_val = cluster_sums[gid];
    float count_f = ((float)(count));
    bool is_empty = (count == 0u);
    float safe_count = ((is_empty) ? (as_type<float>(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_clear_cluster_sums_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_clear_cluster_sums_f32(device const uint* total [[buffer(0)]], device float* cluster_sums [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = total[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    cluster_sums[gid] = as_type<float>(0x00000000u);
  }
}

)kernel";
const char* k_metal_navatala_vector_search_clear_cluster_counts = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_clear_cluster_counts(device const uint* k [[buffer(0)]], device uint* cluster_counts [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    cluster_counts[gid] = 0u;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_reduce_inertia_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_reduce_inertia_f32(device const float* min_distances [[buffer(0)]], device const uint* n_points [[buffer(1)]], device float* partial_sums [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint group_id = ((uint)(int(__tgid.x)));
  uint n = n_points[0];
  threadgroup float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (as_type<float>(0x00000000u)));
  sdata[lid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride = 128u;
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!(stride > 0u)) break;
    uint s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      uint other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stride = (s / 2u);
  }
  bool is_thread0 = (lid == 0u);
  if (is_thread0) {
    float block_sum = sdata[0u];
    partial_sums[group_id] = block_sum;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_bitonic_sort_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_bitonic_sort_step_f32(device const uint* n [[buffer(0)]], device const uint* step [[buffer(1)]], device const uint* stage [[buffer(2)]], device float* keys [[buffer(3)]], device uint* values [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  uint stepVal = step[0];
  uint stageVal = stage[0];
  uint half_n = (nVal / 2u);
  bool inBounds = (gid < half_n);
  if (inBounds) {
    uint distance = (1u << (stepVal - stageVal));
    uint block_size = (2u << stepVal);
    uint block_id = (gid / distance);
    uint offset = (gid % distance);
    uint i = ((block_id * (2u * distance)) + offset);
    uint j = (i + distance);
    uint block_start = ((i / block_size) * block_size);
    bool ascending = (((block_start / block_size) % 2u) == 0u);
    float key_i = keys[i];
    float key_j = keys[j];
    uint val_i = values[i];
    uint val_j = values[j];
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
const char* k_metal_navatala_vector_search_per_row_top_k_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_per_row_top_k_f32(device const float* data [[buffer(0)]], device const uint* n_rows [[buffer(1)]], device const uint* row_size [[buffer(2)]], device const uint* k [[buffer(3)]], device float* top_values [[buffer(4)]], device uint* top_indices [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nr = n_rows[0];
  uint rs = row_size[0];
  uint kVal = k[0];
  uint total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint row_start = (row_idx * rs);
    float best_val = as_type<float>(0x7e967699u);
    uint best_idx = 0u;
    uint count_smaller = 0u;
    for (int i = 0; i < (int)(rs); ++i) {
      uint data_idx = (row_start + i);
      float val = data[data_idx];
      uint smaller_count = 0u;
      for (int j = 0; j < (int)(rs); ++j) {
        uint other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        uint old_cnt = smaller_count;
        uint incr = ((counts) ? (1u) : (0u));
        smaller_count = (old_cnt + incr);
      }
      uint cnt = smaller_count;
      bool is_kth = (cnt == k_idx);
      if (is_kth) {
        best_val = val;
        best_idx = i;
      }
    }
    uint out_idx = ((row_idx * kVal) + k_idx);
    float final_val = best_val;
    uint final_idx = best_idx;
    top_values[out_idx] = final_val;
    top_indices[out_idx] = final_idx;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_merge_sorted_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_merge_sorted_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* a_indices [[buffer(2)]], device const uint* b_indices [[buffer(3)]], device const uint* n_a [[buffer(4)]], device const uint* n_b [[buffer(5)]], device float* merged [[buffer(6)]], device uint* merged_indices [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint na = n_a[0];
  uint nb = n_b[0];
  uint total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint lo = 0u;
    uint hi = gid;
    uint hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!(lo < hi)) break;
      uint lo_v = lo;
      uint hi_v = hi;
      uint mid = (lo_v + ((hi_v - lo_v) / 2u));
      uint j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (as_type<float>(0x7e967699u)));
      float b_prev = (((j > 0u)) ? (b[(j - 1u)]) : (as_type<float>(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + 1u);
      } else {
        hi = mid;
      }
    }
    uint i = lo;
    uint j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (as_type<float>(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (as_type<float>(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    uint out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_rerank_with_exact_distances_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_rerank_with_exact_distances_f32(device const float* queries [[buffer(0)]], device const float* dataset [[buffer(1)]], device const uint* candidate_ids [[buffer(2)]], device const uint* n_queries [[buffer(3)]], device const uint* k [[buffer(4)]], device const uint* dim [[buffer(5)]], device float* exact_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint d = dim[0];
  uint total = (nq * k_val);
  if (gid < total) {
    uint query_id = (gid / k_val);
    uint k_idx = (gid % k_val);
    uint cand_id = candidate_ids[gid];
    float dist = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(d); ++i) {
      uint q_idx = ((query_id * d) + i);
      uint c_idx = ((cand_id * d) + i);
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
const char* k_metal_navatala_vector_search_filter_by_threshold_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_filter_by_threshold_f32(device const float* distances [[buffer(0)]], device uint* candidate_ids [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* k [[buffer(3)]], device const float* threshold [[buffer(4)]], device const uint* invalid_id [[buffer(5)]], device uint* filtered_counts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  float thresh = threshold[0];
  uint inv = invalid_id[0];
  if (query_id < nq) {
    uint count = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      uint idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if (dist > thresh) {
        candidate_ids[idx] = inv;
      } else {
        uint old_cnt = count;
        count = (old_cnt + 1u);
      }
    }
    uint final_count = count;
    filtered_counts[query_id] = final_count;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_merge_search_results_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_merge_search_results_f32(device const uint* ids_a [[buffer(0)]], device const float* dists_a [[buffer(1)]], device const uint* ids_b [[buffer(2)]], device const float* dists_b [[buffer(3)]], device const uint* n_queries [[buffer(4)]], device const uint* k_a [[buffer(5)]], device const uint* k_b [[buffer(6)]], device const uint* k_out [[buffer(7)]], device uint* merged_ids [[buffer(8)]], device float* merged_dists [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint ka = k_a[0];
  uint kb = k_b[0];
  uint ko = k_out[0];
  if (query_id < nq) {
    uint ptr_a = 0u;
    uint ptr_b = 0u;
    uint out_idx = 0u;
    for (int _iter = 0; _iter < (int)(ko); ++_iter) {
      uint pa = ptr_a;
      uint pb = ptr_b;
      uint oi = out_idx;
      bool a_valid = (pa < ka);
      bool b_valid = (pb < kb);
      uint idx_a = ((query_id * ka) + pa);
      uint idx_b = ((query_id * kb) + pb);
      float dist_a_raw = dists_a[idx_a];
      float dist_b_raw = dists_b[idx_b];
      float dist_a = ((a_valid) ? (dist_a_raw) : (as_type<float>(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (as_type<float>(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      uint chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      uint out_flat = ((query_id * ko) + oi);
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
const char* k_metal_navatala_vector_search_compute_recall_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_recall_f32(device const uint* approx_ids [[buffer(0)]], device const uint* ground_truth_ids [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* k [[buffer(3)]], device float* recall_per_query [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint query_id = ((uint)(int(__gid.x)));
  uint nq = n_queries[0];
  uint k_val = k[0];
  if (query_id < nq) {
    uint hits = 0u;
    for (int i = 0; i < (int)(k_val); ++i) {
      uint approx_idx = ((query_id * k_val) + i);
      uint approx_id = approx_ids[approx_idx];
      uint found = 0u;
      for (int j = 0; j < (int)(k_val); ++j) {
        uint gt_idx = ((query_id * k_val) + j);
        uint gt_id = ground_truth_ids[gt_idx];
        if (approx_id == gt_id) {
          found = 1u;
        }
      }
      uint was_found = found;
      uint old_hits = hits;
      hits = (old_hits + was_found);
    }
    uint total_hits = hits;
    float hits_f32 = ((float)(total_hits));
    float k_f32 = ((float)(k_val));
    float recall = (hits_f32 / k_f32);
    recall_per_query[query_id] = recall;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_scalar_quantize_f32_to_i8 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_scalar_quantize_f32_to_i8(device const float* _input [[buffer(0)]], device const uint* n_vectors [[buffer(1)]], device const uint* dim [[buffer(2)]], device const float* mins [[buffer(3)]], device const float* maxs [[buffer(4)]], device char* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if (gid < total) {
    uint dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * as_type<float>(0x437f0000u));
    float offset = (scaled - as_type<float>(0x43000000u));
    float neg128 = (as_type<float>(0x00000000u) - as_type<float>(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > as_type<float>(0x42fe0000u))) ? (as_type<float>(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    char quantized = ((char)(quantized_i32));
    _output[gid] = quantized;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_scalar_dequantize_i8_to_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_scalar_dequantize_i8_to_f32(device const char* _input [[buffer(0)]], device const uint* n_vectors [[buffer(1)]], device const uint* dim [[buffer(2)]], device const float* mins [[buffer(3)]], device const float* maxs [[buffer(4)]], device float* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if (gid < total) {
    uint dim_idx = (gid % d);
    char qval = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float qval_f32 = ((float)(qval));
    float offset = (qval_f32 + as_type<float>(0x43000000u));
    float normalized = (offset / as_type<float>(0x437f0000u));
    float range = (max_val - min_val);
    float dequant = ((normalized * range) + min_val);
    _output[gid] = dequant;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_compute_quantized_distances_i8 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_compute_quantized_distances_i8(device const char* queries [[buffer(0)]], device const char* dataset [[buffer(1)]], device const uint* n_queries [[buffer(2)]], device const uint* n_dataset [[buffer(3)]], device const uint* dim [[buffer(4)]], device const float* scale [[buffer(5)]], device float* distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nq = n_queries[0];
  uint nd = n_dataset[0];
  uint d = dim[0];
  float s = scale[0];
  bool in_x = (gid_x < nq);
  bool in_y = (gid_y < nd);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    int acc = 0;
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((gid_x * d) + k);
      uint d_idx = ((gid_y * d) + k);
      char q_val = queries[q_idx];
      char d_val = dataset[d_idx];
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
    uint out_idx = ((gid_x * nd) + gid_y);
    distances[out_idx] = result;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_encode_p_q_vectors_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_encode_p_q_vectors_f32(device const float* subvectors [[buffer(0)]], device const float* codebook [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* n_codewords [[buffer(3)]], device const uint* sub_dim [[buffer(4)]], device uchar* codes [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint vid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  if (vid < nv) {
    uint best_cw = 0u;
    float best_dist = as_type<float>(0x7e967699u);
    for (int cw = 0; cw < (int)(nc); ++cw) {
      float dist = as_type<float>(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        uint sv_idx = ((vid * sd) + k);
        uint cb_idx = ((cw * sd) + k);
        float sv = subvectors[sv_idx];
        float cb = codebook[cb_idx];
        float diff = (sv - cb);
        float sq = (diff * diff);
        float old_dist = dist;
        dist = (old_dist + sq);
      }
      float cw_dist = dist;
      float curr_best = best_dist;
      if (cw_dist < curr_best) {
        best_cw = cw;
        best_dist = cw_dist;
      }
    }
    uint final_cw = best_cw;
    uchar code = ((uchar)(final_cw));
    codes[vid] = code;
  }
}

)kernel";
const char* k_metal_navatala_vector_search_decode_p_q_vectors_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_decode_p_q_vectors_f32(device const uchar* codes [[buffer(0)]], device const float* codebook [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* sub_dim [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  uint total = (nv * sd);
  if (gid < total) {
    uint vid = (gid / sd);
    uint k = (gid % sd);
    uchar code = codes[vid];
    uint cw = ((uint)(code));
    uint cb_idx = ((cw * sd) + k);
    float val = codebook[cb_idx];
    _output[gid] = val;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_init_seeds_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_init_seeds_f32 = {
  1,
  "navatala_vector_search_init_seeds_f32",
  "metal",
  "navatala_vector_search_init_seeds_f32",
  "kernel:metal:navatala_vector_search_init_seeds_f32",
  "abi-r1:metal:navatala_vector_search_init_seeds_f32",
  "abi-r1:metal:navatala_vector_search_init_seeds_f32",
  9,
  kAbiArgs_metal_navatala_vector_search_init_seeds_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_expand_neighbors_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_expand_neighbors_f32 = {
  1,
  "navatala_vector_search_expand_neighbors_f32",
  "metal",
  "navatala_vector_search_expand_neighbors_f32",
  "kernel:metal:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_expand_neighbors_f32",
  10,
  kAbiArgs_metal_navatala_vector_search_expand_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_check_visited[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "visited", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "hash_size", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "is_new", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_check_visited = {
  1,
  "navatala_vector_search_check_visited",
  "metal",
  "navatala_vector_search_check_visited",
  "kernel:metal:navatala_vector_search_check_visited",
  "abi-r1:metal:navatala_vector_search_check_visited",
  "abi-r1:metal:navatala_vector_search_check_visited",
  5,
  kAbiArgs_metal_navatala_vector_search_check_visited
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_clear_visited[] = {
  { "hash_size", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_clear_visited = {
  1,
  "navatala_vector_search_clear_visited",
  "metal",
  "navatala_vector_search_clear_visited",
  "kernel:metal:navatala_vector_search_clear_visited",
  "abi-r1:metal:navatala_vector_search_clear_visited",
  "abi-r1:metal:navatala_vector_search_clear_visited",
  2,
  kAbiArgs_metal_navatala_vector_search_clear_visited
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_extract_results_f32[] = {
  { "sorted_candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sorted_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "itopk_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_extract_results_f32 = {
  1,
  "navatala_vector_search_extract_results_f32",
  "metal",
  "navatala_vector_search_extract_results_f32",
  "kernel:metal:navatala_vector_search_extract_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_results_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_extract_results_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_distances_batch_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_distances_batch_f32 = {
  1,
  "navatala_vector_search_compute_distances_batch_f32",
  "metal",
  "navatala_vector_search_compute_distances_batch_f32",
  "kernel:metal:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:metal:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:metal:navatala_vector_search_compute_distances_batch_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_compute_distances_batch_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_search_layer_greedy_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_search_layer_greedy_f32 = {
  1,
  "navatala_vector_search_search_layer_greedy_f32",
  "metal",
  "navatala_vector_search_search_layer_greedy_f32",
  "kernel:metal:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:metal:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:metal:navatala_vector_search_search_layer_greedy_f32",
  12,
  kAbiArgs_metal_navatala_vector_search_search_layer_greedy_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_select_neighbors_simple_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selected", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "selected_count", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_select_neighbors_simple_f32 = {
  1,
  "navatala_vector_search_select_neighbors_simple_f32",
  "metal",
  "navatala_vector_search_select_neighbors_simple_f32",
  "kernel:metal:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:metal:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:metal:navatala_vector_search_select_neighbors_simple_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_select_neighbors_simple_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_select_neighbors_heuristic_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_select_neighbors_heuristic_f32 = {
  1,
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "metal",
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "kernel:metal:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:metal:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:metal:navatala_vector_search_select_neighbors_heuristic_f32",
  9,
  kAbiArgs_metal_navatala_vector_search_select_neighbors_heuristic_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_update_candidate_list_f32[] = {
  { "candidates", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_distances", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_current", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_new", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ef", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_merged", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_update_candidate_list_f32 = {
  1,
  "navatala_vector_search_update_candidate_list_f32",
  "metal",
  "navatala_vector_search_update_candidate_list_f32",
  "kernel:metal:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:metal:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:metal:navatala_vector_search_update_candidate_list_f32",
  8,
  kAbiArgs_metal_navatala_vector_search_update_candidate_list_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_extract_layer_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_found", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_extract_layer_results_f32 = {
  1,
  "navatala_vector_search_extract_layer_results_f32",
  "metal",
  "navatala_vector_search_extract_layer_results_f32",
  "kernel:metal:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_layer_results_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_extract_layer_results_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_init_search_state_f32[] = {
  { "global_entry", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_queries", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "entry_points", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_init_search_state_f32 = {
  1,
  "navatala_vector_search_init_search_state_f32",
  "metal",
  "navatala_vector_search_init_search_state_f32",
  "kernel:metal:navatala_vector_search_init_search_state_f32",
  "abi-r1:metal:navatala_vector_search_init_search_state_f32",
  "abi-r1:metal:navatala_vector_search_init_search_state_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_init_search_state_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_mark_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_mark_visited_batch = {
  1,
  "navatala_vector_search_mark_visited_batch",
  "metal",
  "navatala_vector_search_mark_visited_batch",
  "kernel:metal:navatala_vector_search_mark_visited_batch",
  "abi-r1:metal:navatala_vector_search_mark_visited_batch",
  "abi-r1:metal:navatala_vector_search_mark_visited_batch",
  5,
  kAbiArgs_metal_navatala_vector_search_mark_visited_batch
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_check_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "is_visited", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_check_visited_batch = {
  1,
  "navatala_vector_search_check_visited_batch",
  "metal",
  "navatala_vector_search_check_visited_batch",
  "kernel:metal:navatala_vector_search_check_visited_batch",
  "abi-r1:metal:navatala_vector_search_check_visited_batch",
  "abi-r1:metal:navatala_vector_search_check_visited_batch",
  6,
  kAbiArgs_metal_navatala_vector_search_check_visited_batch
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_merge_multi_query_results_f32[] = {
  { "layer_results", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "layer_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_layers", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "final_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "final_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_merge_multi_query_results_f32 = {
  1,
  "navatala_vector_search_merge_multi_query_results_f32",
  "metal",
  "navatala_vector_search_merge_multi_query_results_f32",
  "kernel:metal:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:metal:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:metal:navatala_vector_search_merge_multi_query_results_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_merge_multi_query_results_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_neighbor_distances_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_neighbor_distances_f32 = {
  1,
  "navatala_vector_search_compute_neighbor_distances_f32",
  "metal",
  "navatala_vector_search_compute_neighbor_distances_f32",
  "kernel:metal:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_neighbor_distances_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_compute_neighbor_distances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_greedy_search_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_greedy_search_f32 = {
  1,
  "navatala_vector_search_greedy_search_f32",
  "metal",
  "navatala_vector_search_greedy_search_f32",
  "kernel:metal:navatala_vector_search_greedy_search_f32",
  "abi-r1:metal:navatala_vector_search_greedy_search_f32",
  "abi-r1:metal:navatala_vector_search_greedy_search_f32",
  12,
  kAbiArgs_metal_navatala_vector_search_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_robust_prune_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_robust_prune_f32 = {
  1,
  "navatala_vector_search_robust_prune_f32",
  "metal",
  "navatala_vector_search_robust_prune_f32",
  "kernel:metal:navatala_vector_search_robust_prune_f32",
  "abi-r1:metal:navatala_vector_search_robust_prune_f32",
  "abi-r1:metal:navatala_vector_search_robust_prune_f32",
  10,
  kAbiArgs_metal_navatala_vector_search_robust_prune_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_insert_vertex_f32[] = {
  { "vertex_id", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_insert_vertex_f32 = {
  1,
  "navatala_vector_search_insert_vertex_f32",
  "metal",
  "navatala_vector_search_insert_vertex_f32",
  "kernel:metal:navatala_vector_search_insert_vertex_f32",
  "abi-r1:metal:navatala_vector_search_insert_vertex_f32",
  "abi-r1:metal:navatala_vector_search_insert_vertex_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_insert_vertex_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_random_init_neighbors_f32[] = {
  { "n_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "init_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_random_init_neighbors_f32 = {
  1,
  "navatala_vector_search_random_init_neighbors_f32",
  "metal",
  "navatala_vector_search_random_init_neighbors_f32",
  "kernel:metal:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_random_init_neighbors_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_random_init_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_extract_search_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_dists", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "search_list_size", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_extract_search_results_f32 = {
  1,
  "navatala_vector_search_extract_search_results_f32",
  "metal",
  "navatala_vector_search_extract_search_results_f32",
  "kernel:metal:navatala_vector_search_extract_search_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_search_results_f32",
  "abi-r1:metal:navatala_vector_search_extract_search_results_f32",
  8,
  kAbiArgs_metal_navatala_vector_search_extract_search_results_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_batched_greedy_search_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_batched_greedy_search_f32 = {
  1,
  "navatala_vector_search_batched_greedy_search_f32",
  "metal",
  "navatala_vector_search_batched_greedy_search_f32",
  "kernel:metal:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:metal:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:metal:navatala_vector_search_batched_greedy_search_f32",
  12,
  kAbiArgs_metal_navatala_vector_search_batched_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_centroid_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroid_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_centroid_distances_f32 = {
  1,
  "navatala_vector_search_compute_centroid_distances_f32",
  "metal",
  "navatala_vector_search_compute_centroid_distances_f32",
  "kernel:metal:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_centroid_distances_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_compute_centroid_distances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_scan_inverted_list_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_scan_inverted_list_f32 = {
  1,
  "navatala_vector_search_scan_inverted_list_f32",
  "metal",
  "navatala_vector_search_scan_inverted_list_f32",
  "kernel:metal:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:metal:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:metal:navatala_vector_search_scan_inverted_list_f32",
  12,
  kAbiArgs_metal_navatala_vector_search_scan_inverted_list_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_assign_to_cluster_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_assign_to_cluster_f32 = {
  1,
  "navatala_vector_search_assign_to_cluster_f32",
  "metal",
  "navatala_vector_search_assign_to_cluster_f32",
  "kernel:metal:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:metal:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:metal:navatala_vector_search_assign_to_cluster_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_assign_to_cluster_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_count_cluster_sizes[] = {
  { "labels", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_counts", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_count_cluster_sizes = {
  1,
  "navatala_vector_search_count_cluster_sizes",
  "metal",
  "navatala_vector_search_count_cluster_sizes",
  "kernel:metal:navatala_vector_search_count_cluster_sizes",
  "abi-r1:metal:navatala_vector_search_count_cluster_sizes",
  "abi-r1:metal:navatala_vector_search_count_cluster_sizes",
  4,
  kAbiArgs_metal_navatala_vector_search_count_cluster_sizes
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_populate_inverted_lists_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "list_data", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "list_indices", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "write_positions", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_populate_inverted_lists_f32 = {
  1,
  "navatala_vector_search_populate_inverted_lists_f32",
  "metal",
  "navatala_vector_search_populate_inverted_lists_f32",
  "kernel:metal:navatala_vector_search_populate_inverted_lists_f32",
  "abi-r1:metal:navatala_vector_search_populate_inverted_lists_f32",
  "abi-r1:metal:navatala_vector_search_populate_inverted_lists_f32",
  8,
  kAbiArgs_metal_navatala_vector_search_populate_inverted_lists_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_initialize_random_graph[] = {
  { "n_nodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_initialize_random_graph = {
  1,
  "navatala_vector_search_initialize_random_graph",
  "metal",
  "navatala_vector_search_initialize_random_graph",
  "kernel:metal:navatala_vector_search_initialize_random_graph",
  "abi-r1:metal:navatala_vector_search_initialize_random_graph",
  "abi-r1:metal:navatala_vector_search_initialize_random_graph",
  5,
  kAbiArgs_metal_navatala_vector_search_initialize_random_graph
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_graph_distances_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_graph_distances_f32 = {
  1,
  "navatala_vector_search_compute_graph_distances_f32",
  "metal",
  "navatala_vector_search_compute_graph_distances_f32",
  "kernel:metal:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:metal:navatala_vector_search_compute_graph_distances_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_compute_graph_distances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_nn_descent_join_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_nn_descent_join_f32 = {
  1,
  "navatala_vector_search_nn_descent_join_f32",
  "metal",
  "navatala_vector_search_nn_descent_join_f32",
  "kernel:metal:navatala_vector_search_nn_descent_join_f32",
  "abi-r1:metal:navatala_vector_search_nn_descent_join_f32",
  "abi-r1:metal:navatala_vector_search_nn_descent_join_f32",
  9,
  kAbiArgs_metal_navatala_vector_search_nn_descent_join_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_build_reverse_graph[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "reverse_graph", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "reverse_counts", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_build_reverse_graph = {
  1,
  "navatala_vector_search_build_reverse_graph",
  "metal",
  "navatala_vector_search_build_reverse_graph",
  "kernel:metal:navatala_vector_search_build_reverse_graph",
  "abi-r1:metal:navatala_vector_search_build_reverse_graph",
  "abi-r1:metal:navatala_vector_search_build_reverse_graph",
  5,
  kAbiArgs_metal_navatala_vector_search_build_reverse_graph
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_prune_graph_r_n_g_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pruned_graph", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_prune_graph_r_n_g_f32 = {
  1,
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "metal",
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "kernel:metal:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:metal:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:metal:navatala_vector_search_prune_graph_r_n_g_f32",
  8,
  kAbiArgs_metal_navatala_vector_search_prune_graph_r_n_g_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compact_pruned_graph[] = {
  { "pruned_graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "compact_graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compact_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compact_pruned_graph = {
  1,
  "navatala_vector_search_compact_pruned_graph",
  "metal",
  "navatala_vector_search_compact_pruned_graph",
  "kernel:metal:navatala_vector_search_compact_pruned_graph",
  "abi-r1:metal:navatala_vector_search_compact_pruned_graph",
  "abi-r1:metal:navatala_vector_search_compact_pruned_graph",
  7,
  kAbiArgs_metal_navatala_vector_search_compact_pruned_graph
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_vertex_degrees[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_vertex_degrees = {
  1,
  "navatala_vector_search_compute_vertex_degrees",
  "metal",
  "navatala_vector_search_compute_vertex_degrees",
  "kernel:metal:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:metal:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:metal:navatala_vector_search_compute_vertex_degrees",
  5,
  kAbiArgs_metal_navatala_vector_search_compute_vertex_degrees
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_prune_excess_degree[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_prune_excess_degree = {
  1,
  "navatala_vector_search_prune_excess_degree",
  "metal",
  "navatala_vector_search_prune_excess_degree",
  "kernel:metal:navatala_vector_search_prune_excess_degree",
  "abi-r1:metal:navatala_vector_search_prune_excess_degree",
  "abi-r1:metal:navatala_vector_search_prune_excess_degree",
  7,
  kAbiArgs_metal_navatala_vector_search_prune_excess_degree
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_reorder_vertices_by_degree[] = {
  { "degrees", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prefix_sums", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_bins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "old_to_new", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_to_old", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_reorder_vertices_by_degree = {
  1,
  "navatala_vector_search_reorder_vertices_by_degree",
  "metal",
  "navatala_vector_search_reorder_vertices_by_degree",
  "kernel:metal:navatala_vector_search_reorder_vertices_by_degree",
  "abi-r1:metal:navatala_vector_search_reorder_vertices_by_degree",
  "abi-r1:metal:navatala_vector_search_reorder_vertices_by_degree",
  6,
  kAbiArgs_metal_navatala_vector_search_reorder_vertices_by_degree
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compact_graph_edges[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compact_graph_edges = {
  1,
  "navatala_vector_search_compact_graph_edges",
  "metal",
  "navatala_vector_search_compact_graph_edges",
  "kernel:metal:navatala_vector_search_compact_graph_edges",
  "abi-r1:metal:navatala_vector_search_compact_graph_edges",
  "abi-r1:metal:navatala_vector_search_compact_graph_edges",
  5,
  kAbiArgs_metal_navatala_vector_search_compact_graph_edges
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_validate_graph_integrity[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "error_flags", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_validate_graph_integrity = {
  1,
  "navatala_vector_search_validate_graph_integrity",
  "metal",
  "navatala_vector_search_validate_graph_integrity",
  "kernel:metal:navatala_vector_search_validate_graph_integrity",
  "abi-r1:metal:navatala_vector_search_validate_graph_integrity",
  "abi-r1:metal:navatala_vector_search_validate_graph_integrity",
  5,
  kAbiArgs_metal_navatala_vector_search_validate_graph_integrity
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_prune_excess_degree_sorted[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_prune_excess_degree_sorted = {
  1,
  "navatala_vector_search_prune_excess_degree_sorted",
  "metal",
  "navatala_vector_search_prune_excess_degree_sorted",
  "kernel:metal:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:metal:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:metal:navatala_vector_search_prune_excess_degree_sorted",
  7,
  kAbiArgs_metal_navatala_vector_search_prune_excess_degree_sorted
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_accumulate_p_q_codebook_i32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook_sums_i32", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_accumulate_p_q_codebook_i32 = {
  1,
  "navatala_vector_search_accumulate_p_q_codebook_i32",
  "metal",
  "navatala_vector_search_accumulate_p_q_codebook_i32",
  "kernel:metal:navatala_vector_search_accumulate_p_q_codebook_i32",
  "abi-r1:metal:navatala_vector_search_accumulate_p_q_codebook_i32",
  "abi-r1:metal:navatala_vector_search_accumulate_p_q_codebook_i32",
  6,
  kAbiArgs_metal_navatala_vector_search_accumulate_p_q_codebook_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_normalize_p_q_codebook_f32[] = {
  { "codebook_sums_i32", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_codewords", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_normalize_p_q_codebook_f32 = {
  1,
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "metal",
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "kernel:metal:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:metal:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:metal:navatala_vector_search_normalize_p_q_codebook_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_normalize_p_q_codebook_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_train_p_q_codebook_f32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "assignments", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codewords", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook_sums", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_train_p_q_codebook_f32 = {
  1,
  "navatala_vector_search_train_p_q_codebook_f32",
  "metal",
  "navatala_vector_search_train_p_q_codebook_f32",
  "kernel:metal:navatala_vector_search_train_p_q_codebook_f32",
  "abi-r1:metal:navatala_vector_search_train_p_q_codebook_f32",
  "abi-r1:metal:navatala_vector_search_train_p_q_codebook_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_train_p_q_codebook_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_l2_squared_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_l2_squared_f32 = {
  1,
  "navatala_vector_search_l2_squared_f32",
  "metal",
  "navatala_vector_search_l2_squared_f32",
  "kernel:metal:navatala_vector_search_l2_squared_f32",
  "abi-r1:metal:navatala_vector_search_l2_squared_f32",
  "abi-r1:metal:navatala_vector_search_l2_squared_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_l2_squared_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_inner_product_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_inner_product_f32 = {
  1,
  "navatala_vector_search_inner_product_f32",
  "metal",
  "navatala_vector_search_inner_product_f32",
  "kernel:metal:navatala_vector_search_inner_product_f32",
  "abi-r1:metal:navatala_vector_search_inner_product_f32",
  "abi-r1:metal:navatala_vector_search_inner_product_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_l1_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_l1_distance_f32 = {
  1,
  "navatala_vector_search_l1_distance_f32",
  "metal",
  "navatala_vector_search_l1_distance_f32",
  "kernel:metal:navatala_vector_search_l1_distance_f32",
  "abi-r1:metal:navatala_vector_search_l1_distance_f32",
  "abi-r1:metal:navatala_vector_search_l1_distance_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_l1_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_batch_l2_to_neighbors_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_batch_l2_to_neighbors_f32 = {
  1,
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "metal",
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "kernel:metal:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:metal:navatala_vector_search_batch_l2_to_neighbors_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_batch_l2_to_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_l2_squared_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_l2_squared_f16 = {
  1,
  "navatala_vector_search_l2_squared_f16",
  "metal",
  "navatala_vector_search_l2_squared_f16",
  "kernel:metal:navatala_vector_search_l2_squared_f16",
  "abi-r1:metal:navatala_vector_search_l2_squared_f16",
  "abi-r1:metal:navatala_vector_search_l2_squared_f16",
  6,
  kAbiArgs_metal_navatala_vector_search_l2_squared_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_cosine_similarity_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sim", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_cosine_similarity_f32 = {
  1,
  "navatala_vector_search_cosine_similarity_f32",
  "metal",
  "navatala_vector_search_cosine_similarity_f32",
  "kernel:metal:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:metal:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:metal:navatala_vector_search_cosine_similarity_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_cosine_similarity_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_linf_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_linf_distance_f32 = {
  1,
  "navatala_vector_search_linf_distance_f32",
  "metal",
  "navatala_vector_search_linf_distance_f32",
  "kernel:metal:navatala_vector_search_linf_distance_f32",
  "abi-r1:metal:navatala_vector_search_linf_distance_f32",
  "abi-r1:metal:navatala_vector_search_linf_distance_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_linf_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_hamming_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_hamming_distance_u32 = {
  1,
  "navatala_vector_search_hamming_distance_u32",
  "metal",
  "navatala_vector_search_hamming_distance_u32",
  "kernel:metal:navatala_vector_search_hamming_distance_u32",
  "abi-r1:metal:navatala_vector_search_hamming_distance_u32",
  "abi-r1:metal:navatala_vector_search_hamming_distance_u32",
  6,
  kAbiArgs_metal_navatala_vector_search_hamming_distance_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_correlation_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_correlation_distance_f32 = {
  1,
  "navatala_vector_search_correlation_distance_f32",
  "metal",
  "navatala_vector_search_correlation_distance_f32",
  "kernel:metal:navatala_vector_search_correlation_distance_f32",
  "abi-r1:metal:navatala_vector_search_correlation_distance_f32",
  "abi-r1:metal:navatala_vector_search_correlation_distance_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_correlation_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_jaccard_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_jaccard_distance_u32 = {
  1,
  "navatala_vector_search_jaccard_distance_u32",
  "metal",
  "navatala_vector_search_jaccard_distance_u32",
  "kernel:metal:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:metal:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:metal:navatala_vector_search_jaccard_distance_u32",
  6,
  kAbiArgs_metal_navatala_vector_search_jaccard_distance_u32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_assign_labels_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "min_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_assign_labels_f32 = {
  1,
  "navatala_vector_search_assign_labels_f32",
  "metal",
  "navatala_vector_search_assign_labels_f32",
  "kernel:metal:navatala_vector_search_assign_labels_f32",
  "abi-r1:metal:navatala_vector_search_assign_labels_f32",
  "abi-r1:metal:navatala_vector_search_assign_labels_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_assign_labels_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_accumulate_cluster_sums_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "labels", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_sums", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_counts", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_accumulate_cluster_sums_f32 = {
  1,
  "navatala_vector_search_accumulate_cluster_sums_f32",
  "metal",
  "navatala_vector_search_accumulate_cluster_sums_f32",
  "kernel:metal:navatala_vector_search_accumulate_cluster_sums_f32",
  "abi-r1:metal:navatala_vector_search_accumulate_cluster_sums_f32",
  "abi-r1:metal:navatala_vector_search_accumulate_cluster_sums_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_accumulate_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_update_centroids_f32[] = {
  { "cluster_sums", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroids", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_update_centroids_f32 = {
  1,
  "navatala_vector_search_update_centroids_f32",
  "metal",
  "navatala_vector_search_update_centroids_f32",
  "kernel:metal:navatala_vector_search_update_centroids_f32",
  "abi-r1:metal:navatala_vector_search_update_centroids_f32",
  "abi-r1:metal:navatala_vector_search_update_centroids_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_update_centroids_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_clear_cluster_sums_f32[] = {
  { "total", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_sums", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_clear_cluster_sums_f32 = {
  1,
  "navatala_vector_search_clear_cluster_sums_f32",
  "metal",
  "navatala_vector_search_clear_cluster_sums_f32",
  "kernel:metal:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:metal:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:metal:navatala_vector_search_clear_cluster_sums_f32",
  2,
  kAbiArgs_metal_navatala_vector_search_clear_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_clear_cluster_counts[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_clear_cluster_counts = {
  1,
  "navatala_vector_search_clear_cluster_counts",
  "metal",
  "navatala_vector_search_clear_cluster_counts",
  "kernel:metal:navatala_vector_search_clear_cluster_counts",
  "abi-r1:metal:navatala_vector_search_clear_cluster_counts",
  "abi-r1:metal:navatala_vector_search_clear_cluster_counts",
  2,
  kAbiArgs_metal_navatala_vector_search_clear_cluster_counts
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_reduce_inertia_f32[] = {
  { "min_distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partial_sums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_reduce_inertia_f32 = {
  1,
  "navatala_vector_search_reduce_inertia_f32",
  "metal",
  "navatala_vector_search_reduce_inertia_f32",
  "kernel:metal:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:metal:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:metal:navatala_vector_search_reduce_inertia_f32",
  3,
  kAbiArgs_metal_navatala_vector_search_reduce_inertia_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_bitonic_sort_step_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "step", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stage", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_bitonic_sort_step_f32 = {
  1,
  "navatala_vector_search_bitonic_sort_step_f32",
  "metal",
  "navatala_vector_search_bitonic_sort_step_f32",
  "kernel:metal:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:metal:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:metal:navatala_vector_search_bitonic_sort_step_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_bitonic_sort_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_per_row_top_k_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "row_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "top_values", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "top_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_per_row_top_k_f32 = {
  1,
  "navatala_vector_search_per_row_top_k_f32",
  "metal",
  "navatala_vector_search_per_row_top_k_f32",
  "kernel:metal:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:metal:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:metal:navatala_vector_search_per_row_top_k_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_per_row_top_k_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_merge_sorted_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "a_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b_indices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_a", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_b", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "merged", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "merged_indices", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_merge_sorted_f32 = {
  1,
  "navatala_vector_search_merge_sorted_f32",
  "metal",
  "navatala_vector_search_merge_sorted_f32",
  "kernel:metal:navatala_vector_search_merge_sorted_f32",
  "abi-r1:metal:navatala_vector_search_merge_sorted_f32",
  "abi-r1:metal:navatala_vector_search_merge_sorted_f32",
  8,
  kAbiArgs_metal_navatala_vector_search_merge_sorted_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_rerank_with_exact_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "exact_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_rerank_with_exact_distances_f32 = {
  1,
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "metal",
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "kernel:metal:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:metal:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:metal:navatala_vector_search_rerank_with_exact_distances_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_rerank_with_exact_distances_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_filter_by_threshold_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "threshold", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "filtered_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_filter_by_threshold_f32 = {
  1,
  "navatala_vector_search_filter_by_threshold_f32",
  "metal",
  "navatala_vector_search_filter_by_threshold_f32",
  "kernel:metal:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:metal:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:metal:navatala_vector_search_filter_by_threshold_f32",
  7,
  kAbiArgs_metal_navatala_vector_search_filter_by_threshold_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_merge_search_results_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_merge_search_results_f32 = {
  1,
  "navatala_vector_search_merge_search_results_f32",
  "metal",
  "navatala_vector_search_merge_search_results_f32",
  "kernel:metal:navatala_vector_search_merge_search_results_f32",
  "abi-r1:metal:navatala_vector_search_merge_search_results_f32",
  "abi-r1:metal:navatala_vector_search_merge_search_results_f32",
  10,
  kAbiArgs_metal_navatala_vector_search_merge_search_results_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_recall_f32[] = {
  { "approx_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ground_truth_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recall_per_query", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_recall_f32 = {
  1,
  "navatala_vector_search_compute_recall_f32",
  "metal",
  "navatala_vector_search_compute_recall_f32",
  "kernel:metal:navatala_vector_search_compute_recall_f32",
  "abi-r1:metal:navatala_vector_search_compute_recall_f32",
  "abi-r1:metal:navatala_vector_search_compute_recall_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_compute_recall_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_scalar_quantize_f32_to_i8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_scalar_quantize_f32_to_i8 = {
  1,
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "metal",
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "kernel:metal:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:metal:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:metal:navatala_vector_search_scalar_quantize_f32_to_i8",
  6,
  kAbiArgs_metal_navatala_vector_search_scalar_quantize_f32_to_i8
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_scalar_dequantize_i8_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_scalar_dequantize_i8_to_f32 = {
  1,
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "metal",
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "kernel:metal:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:metal:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:metal:navatala_vector_search_scalar_dequantize_i8_to_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_scalar_dequantize_i8_to_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_compute_quantized_distances_i8[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_dataset", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scale", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_compute_quantized_distances_i8 = {
  1,
  "navatala_vector_search_compute_quantized_distances_i8",
  "metal",
  "navatala_vector_search_compute_quantized_distances_i8",
  "kernel:metal:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:metal:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:metal:navatala_vector_search_compute_quantized_distances_i8",
  7,
  kAbiArgs_metal_navatala_vector_search_compute_quantized_distances_i8
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_encode_p_q_vectors_f32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codewords", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_encode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_encode_p_q_vectors_f32",
  "metal",
  "navatala_vector_search_encode_p_q_vectors_f32",
  "kernel:metal:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:metal:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:metal:navatala_vector_search_encode_p_q_vectors_f32",
  6,
  kAbiArgs_metal_navatala_vector_search_encode_p_q_vectors_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_vector_search_decode_p_q_vectors_f32[] = {
  { "codes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_vector_search_decode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_decode_p_q_vectors_f32",
  "metal",
  "navatala_vector_search_decode_p_q_vectors_f32",
  "kernel:metal:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:metal:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:metal:navatala_vector_search_decode_p_q_vectors_f32",
  5,
  kAbiArgs_metal_navatala_vector_search_decode_p_q_vectors_f32
};

bool tryGetKernelAbiManifest_metal_vector_search(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_init_seeds_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_expand_neighbors_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_check_visited") {
    out = &kAbiManifest_metal_navatala_vector_search_check_visited;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_visited") {
    out = &kAbiManifest_metal_navatala_vector_search_clear_visited;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_results_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_extract_results_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_distances_batch_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_search_layer_greedy_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_select_neighbors_simple_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_select_neighbors_heuristic_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_update_candidate_list_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_extract_layer_results_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_init_search_state_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out = &kAbiManifest_metal_navatala_vector_search_mark_visited_batch;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_check_visited_batch") {
    out = &kAbiManifest_metal_navatala_vector_search_check_visited_batch;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_merge_multi_query_results_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_neighbor_distances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_greedy_search_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_robust_prune_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_insert_vertex_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_random_init_neighbors_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_extract_search_results_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_batched_greedy_search_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_centroid_distances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_scan_inverted_list_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_assign_to_cluster_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_count_cluster_sizes") {
    out = &kAbiManifest_metal_navatala_vector_search_count_cluster_sizes;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_populate_inverted_lists_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_populate_inverted_lists_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out = &kAbiManifest_metal_navatala_vector_search_initialize_random_graph;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_graph_distances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_nn_descent_join_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_nn_descent_join_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_build_reverse_graph") {
    out = &kAbiManifest_metal_navatala_vector_search_build_reverse_graph;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_prune_graph_r_n_g_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out = &kAbiManifest_metal_navatala_vector_search_compact_pruned_graph;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_vertex_degrees;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out = &kAbiManifest_metal_navatala_vector_search_prune_excess_degree;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_reorder_vertices_by_degree") {
    out = &kAbiManifest_metal_navatala_vector_search_reorder_vertices_by_degree;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out = &kAbiManifest_metal_navatala_vector_search_compact_graph_edges;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out = &kAbiManifest_metal_navatala_vector_search_validate_graph_integrity;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out = &kAbiManifest_metal_navatala_vector_search_prune_excess_degree_sorted;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_accumulate_p_q_codebook_i32") {
    out = &kAbiManifest_metal_navatala_vector_search_accumulate_p_q_codebook_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_normalize_p_q_codebook_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_train_p_q_codebook_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_train_p_q_codebook_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_l2_squared_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_inner_product_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_inner_product_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_l1_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_batch_l2_to_neighbors_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out = &kAbiManifest_metal_navatala_vector_search_l2_squared_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_cosine_similarity_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_linf_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out = &kAbiManifest_metal_navatala_vector_search_hamming_distance_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_correlation_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out = &kAbiManifest_metal_navatala_vector_search_jaccard_distance_u32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_assign_labels_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_accumulate_cluster_sums_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_accumulate_cluster_sums_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_update_centroids_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_clear_cluster_sums_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out = &kAbiManifest_metal_navatala_vector_search_clear_cluster_counts;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_reduce_inertia_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_bitonic_sort_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_per_row_top_k_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_merge_sorted_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_rerank_with_exact_distances_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_filter_by_threshold_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_merge_search_results_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_recall_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out = &kAbiManifest_metal_navatala_vector_search_scalar_quantize_f32_to_i8;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_scalar_dequantize_i8_to_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out = &kAbiManifest_metal_navatala_vector_search_compute_quantized_distances_i8;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_encode_p_q_vectors_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out = &kAbiManifest_metal_navatala_vector_search_decode_p_q_vectors_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_vector_search(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_init_seeds_f32";
    std::string_view sv(k_metal_navatala_vector_search_init_seeds_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_expand_neighbors_f32";
    std::string_view sv(k_metal_navatala_vector_search_expand_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_check_visited") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_check_visited";
    std::string_view sv(k_metal_navatala_vector_search_check_visited);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_visited") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_clear_visited";
    std::string_view sv(k_metal_navatala_vector_search_clear_visited);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_extract_results_f32";
    std::string_view sv(k_metal_navatala_vector_search_extract_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_distances_batch_f32";
    std::string_view sv(k_metal_navatala_vector_search_compute_distances_batch_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_search_layer_greedy_f32";
    std::string_view sv(k_metal_navatala_vector_search_search_layer_greedy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_select_neighbors_simple_f32";
    std::string_view sv(k_metal_navatala_vector_search_select_neighbors_simple_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_select_neighbors_heuristic_f32";
    std::string_view sv(k_metal_navatala_vector_search_select_neighbors_heuristic_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_update_candidate_list_f32";
    std::string_view sv(k_metal_navatala_vector_search_update_candidate_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_extract_layer_results_f32";
    std::string_view sv(k_metal_navatala_vector_search_extract_layer_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_init_search_state_f32";
    std::string_view sv(k_metal_navatala_vector_search_init_search_state_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_mark_visited_batch";
    std::string_view sv(k_metal_navatala_vector_search_mark_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_check_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_check_visited_batch";
    std::string_view sv(k_metal_navatala_vector_search_check_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_merge_multi_query_results_f32";
    std::string_view sv(k_metal_navatala_vector_search_merge_multi_query_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_neighbor_distances_f32";
    std::string_view sv(k_metal_navatala_vector_search_compute_neighbor_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_greedy_search_f32";
    std::string_view sv(k_metal_navatala_vector_search_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_robust_prune_f32";
    std::string_view sv(k_metal_navatala_vector_search_robust_prune_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_insert_vertex_f32";
    std::string_view sv(k_metal_navatala_vector_search_insert_vertex_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_random_init_neighbors_f32";
    std::string_view sv(k_metal_navatala_vector_search_random_init_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_extract_search_results_f32";
    std::string_view sv(k_metal_navatala_vector_search_extract_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_batched_greedy_search_f32";
    std::string_view sv(k_metal_navatala_vector_search_batched_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_centroid_distances_f32";
    std::string_view sv(k_metal_navatala_vector_search_compute_centroid_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_scan_inverted_list_f32";
    std::string_view sv(k_metal_navatala_vector_search_scan_inverted_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_assign_to_cluster_f32";
    std::string_view sv(k_metal_navatala_vector_search_assign_to_cluster_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_count_cluster_sizes") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_count_cluster_sizes";
    std::string_view sv(k_metal_navatala_vector_search_count_cluster_sizes);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_populate_inverted_lists_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_populate_inverted_lists_f32";
    std::string_view sv(k_metal_navatala_vector_search_populate_inverted_lists_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_initialize_random_graph";
    std::string_view sv(k_metal_navatala_vector_search_initialize_random_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_graph_distances_f32";
    std::string_view sv(k_metal_navatala_vector_search_compute_graph_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_nn_descent_join_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_nn_descent_join_f32";
    std::string_view sv(k_metal_navatala_vector_search_nn_descent_join_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_build_reverse_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_build_reverse_graph";
    std::string_view sv(k_metal_navatala_vector_search_build_reverse_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_prune_graph_r_n_g_f32";
    std::string_view sv(k_metal_navatala_vector_search_prune_graph_r_n_g_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compact_pruned_graph";
    std::string_view sv(k_metal_navatala_vector_search_compact_pruned_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_vertex_degrees";
    std::string_view sv(k_metal_navatala_vector_search_compute_vertex_degrees);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_prune_excess_degree";
    std::string_view sv(k_metal_navatala_vector_search_prune_excess_degree);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_reorder_vertices_by_degree") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_reorder_vertices_by_degree";
    std::string_view sv(k_metal_navatala_vector_search_reorder_vertices_by_degree);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compact_graph_edges";
    std::string_view sv(k_metal_navatala_vector_search_compact_graph_edges);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_validate_graph_integrity";
    std::string_view sv(k_metal_navatala_vector_search_validate_graph_integrity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_prune_excess_degree_sorted";
    std::string_view sv(k_metal_navatala_vector_search_prune_excess_degree_sorted);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_accumulate_p_q_codebook_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_accumulate_p_q_codebook_i32";
    std::string_view sv(k_metal_navatala_vector_search_accumulate_p_q_codebook_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_normalize_p_q_codebook_f32";
    std::string_view sv(k_metal_navatala_vector_search_normalize_p_q_codebook_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_train_p_q_codebook_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_train_p_q_codebook_f32";
    std::string_view sv(k_metal_navatala_vector_search_train_p_q_codebook_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_l2_squared_f32";
    std::string_view sv(k_metal_navatala_vector_search_l2_squared_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_inner_product_f32";
    std::string_view sv(k_metal_navatala_vector_search_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_l1_distance_f32";
    std::string_view sv(k_metal_navatala_vector_search_l1_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_batch_l2_to_neighbors_f32";
    std::string_view sv(k_metal_navatala_vector_search_batch_l2_to_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_l2_squared_f16";
    std::string_view sv(k_metal_navatala_vector_search_l2_squared_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_cosine_similarity_f32";
    std::string_view sv(k_metal_navatala_vector_search_cosine_similarity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_linf_distance_f32";
    std::string_view sv(k_metal_navatala_vector_search_linf_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_hamming_distance_u32";
    std::string_view sv(k_metal_navatala_vector_search_hamming_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_correlation_distance_f32";
    std::string_view sv(k_metal_navatala_vector_search_correlation_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_jaccard_distance_u32";
    std::string_view sv(k_metal_navatala_vector_search_jaccard_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_assign_labels_f32";
    std::string_view sv(k_metal_navatala_vector_search_assign_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_accumulate_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_accumulate_cluster_sums_f32";
    std::string_view sv(k_metal_navatala_vector_search_accumulate_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_update_centroids_f32";
    std::string_view sv(k_metal_navatala_vector_search_update_centroids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_clear_cluster_sums_f32";
    std::string_view sv(k_metal_navatala_vector_search_clear_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_clear_cluster_counts";
    std::string_view sv(k_metal_navatala_vector_search_clear_cluster_counts);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_reduce_inertia_f32";
    std::string_view sv(k_metal_navatala_vector_search_reduce_inertia_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_bitonic_sort_step_f32";
    std::string_view sv(k_metal_navatala_vector_search_bitonic_sort_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_per_row_top_k_f32";
    std::string_view sv(k_metal_navatala_vector_search_per_row_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_merge_sorted_f32";
    std::string_view sv(k_metal_navatala_vector_search_merge_sorted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_rerank_with_exact_distances_f32";
    std::string_view sv(k_metal_navatala_vector_search_rerank_with_exact_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_filter_by_threshold_f32";
    std::string_view sv(k_metal_navatala_vector_search_filter_by_threshold_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_merge_search_results_f32";
    std::string_view sv(k_metal_navatala_vector_search_merge_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_recall_f32";
    std::string_view sv(k_metal_navatala_vector_search_compute_recall_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_scalar_quantize_f32_to_i8";
    std::string_view sv(k_metal_navatala_vector_search_scalar_quantize_f32_to_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_scalar_dequantize_i8_to_f32";
    std::string_view sv(k_metal_navatala_vector_search_scalar_dequantize_i8_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_compute_quantized_distances_i8";
    std::string_view sv(k_metal_navatala_vector_search_compute_quantized_distances_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_encode_p_q_vectors_f32";
    std::string_view sv(k_metal_navatala_vector_search_encode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_vector_search_decode_p_q_vectors_f32";
    std::string_view sv(k_metal_navatala_vector_search_decode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

