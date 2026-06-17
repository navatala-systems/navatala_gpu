// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_vector_search` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_vector_search_init_seeds_f32 = R"kernel(
__kernel void navatala_vector_search_init_seeds_f32(__global const float* queries, __global const float* dataset, __global const uint* seed_indices, __global const uint* n_queries, __global const uint* n_seeds, __global const uint* dim, __global const uint* itopk_size, __global uint* candidates, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    float dist_acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_expand_neighbors_f32 = R"kernel(
__kernel void navatala_vector_search_expand_neighbors_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* top_candidates, __global const uint* n_queries, __global const uint* search_width, __global const uint* graph_degree, __global const uint* dim, __global uint* new_candidates, __global float* new_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    float dist_acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_clear_visited = R"kernel(
__kernel void navatala_vector_search_clear_visited(__global const uint* hash_size, __global uint* visited) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint hs = hash_size[0];
  bool inBounds = (gid < hs);
  if (inBounds) {
    visited[gid] = (uint)(0u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_extract_results_f32 = R"kernel(
__kernel void navatala_vector_search_extract_results_f32(__global const uint* sorted_candidates, __global const float* sorted_distances, __global const uint* n_queries, __global const uint* k, __global const uint* itopk_size, __global uint* result_indices, __global float* result_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
const char* k_opencl_navatala_vector_search_compute_distances_batch_f32 = R"kernel(
__kernel void navatala_vector_search_compute_distances_batch_f32(__global const float* query, __global const float* candidates, __global const uint* candidate_ids, __global const uint* n_candidates, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n_cand = n_candidates[0];
  uint d = dim[0];
  if ((gid < n_cand)) {
    uint cand_id = candidate_ids[gid];
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_search_layer_greedy_f32 = R"kernel(
__kernel void navatala_vector_search_search_layer_greedy_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_points, __global const uint* n_queries, __global const uint* dim, __global const uint* ef, __global const uint* max_degree, __global uint* candidates, __global float* candidate_distances, __global uint* n_found) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  if ((query_id < nq)) {
    uint entry = entry_points[query_id];
    float best_dist = as_float(0x00000000u);
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
    n_found[query_id] = (uint)(1u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_select_neighbors_simple_f32 = R"kernel(
__kernel void navatala_vector_search_select_neighbors_simple_f32(__global const uint* candidates, __global const float* distances, __global const uint* n_candidates, __global const uint* m, __global uint* selected, __global uint* selected_count) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint n_cand = n_candidates[0];
  uint m_val = m[0];
  if ((tid == (uint)(0u))) {
    uint limit = (((n_cand < m_val)) ? (n_cand) : (m_val));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    selected_count[(uint)(0u)] = limit;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_select_neighbors_heuristic_f32 = R"kernel(
__kernel void navatala_vector_search_select_neighbors_heuristic_f32(__global const float* query, __global const uint* candidates, __global const float* distances, __global const float* dataset, __global const uint* n_candidates, __global const uint* dim, __global const uint* m, __global uint* selected, __global uint* n_selected) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint n_cand = n_candidates[0];
  uint m_val = m[0];
  __local uint selected_ids[64];
  __local uint n_sel[1];
  if ((tid == (uint)(0u))) {
    n_sel[(uint)(0u)] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid < n_cand)) {
    uint cand_id = candidates[tid];
    selected[tid] = cand_id;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid == (uint)(0u))) {
    uint final_count = (((n_cand < m_val)) ? (n_cand) : (m_val));
    n_selected[(uint)(0u)] = final_count;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_update_candidate_list_f32 = R"kernel(
__kernel void navatala_vector_search_update_candidate_list_f32(__global uint* candidates, __global float* distances, __global const uint* new_candidates, __global const float* new_distances, __global const uint* n_current, __global const uint* n_new, __global const uint* ef, __global uint* n_merged) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint n_curr = n_current[0];
  uint n_new_val = n_new[0];
  uint ef_val = ef[0];
  __local uint merge_ids[512];
  __local float merge_dists[512];
  if ((tid < n_curr)) {
    uint cid = candidates[tid];
    float cdist = distances[tid];
    merge_ids[tid] = cid;
    merge_dists[tid] = cdist;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid < n_new_val)) {
    uint offset = (n_curr + tid);
    uint nid = new_candidates[tid];
    float ndist = new_distances[tid];
    merge_ids[offset] = nid;
    merge_dists[offset] = ndist;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((tid == (uint)(0u))) {
    uint total = (n_curr + n_new_val);
    uint final_n = (((total < ef_val)) ? (total) : (ef_val));
    n_merged[(uint)(0u)] = final_n;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_extract_layer_results_f32 = R"kernel(
__kernel void navatala_vector_search_extract_layer_results_f32(__global const uint* candidates, __global const float* distances, __global const uint* n_found, __global const uint* n_queries, __global const uint* k, __global uint* result_ids, __global float* result_dists) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint query_id = (gid / k_val);
  uint neighbor_idx = (gid % k_val);
  if ((query_id < nq)) {
    uint found = n_found[query_id];
    uint src_idx = ((query_id * k_val) + neighbor_idx);
    if ((neighbor_idx < found)) {
      uint cand_id = candidates[src_idx];
      float cand_dist = distances[src_idx];
      result_ids[src_idx] = cand_id;
      result_dists[src_idx] = cand_dist;
    } else {
      result_ids[src_idx] = (uint)(4294967295u);
      result_dists[src_idx] = as_float(0x7e967699u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_init_search_state_f32 = R"kernel(
__kernel void navatala_vector_search_init_search_state_f32(__global const uint* global_entry, __global const uint* n_queries, __global const uint* visited_size, __global uint* entry_points, __global uint* visited) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint entry = global_entry[0];
  uint nq = n_queries[0];
  uint vs = visited_size[0];
  if ((gid < nq)) {
    entry_points[gid] = entry;
  }
  uint total_visited = (nq * vs);
  if ((gid < total_visited)) {
    uint q_idx = (gid / vs);
    uint v_idx = (gid % vs);
    uint flat_idx = ((q_idx * vs) + v_idx);
    visited[flat_idx] = (uint)(0u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_mark_visited_batch = R"kernel(
__kernel void navatala_vector_search_mark_visited_batch(__global const uint* node_ids, __global const uint* query_ids, __global const uint* n_nodes, __global const uint* visited_stride, __global uint* visited) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if ((tid < nn)) {
    uint node = node_ids[tid];
    uint query = query_ids[tid];
    uint word_idx = (node / (uint)(32u));
    uint bit_idx = (node % (uint)(32u));
    uint bit_mask = ((uint)(1u) << bit_idx);
    uint flat_idx = ((query * stride) + word_idx);
    atomic_or(&visited[flat_idx], bit_mask);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_check_visited_batch = R"kernel(
__kernel void navatala_vector_search_check_visited_batch(__global const uint* node_ids, __global const uint* query_ids, __global const uint* n_nodes, __global const uint* visited_stride, __global const uint* visited, __global uint* is_visited) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if ((tid < nn)) {
    uint node = node_ids[tid];
    uint query = query_ids[tid];
    uint word_idx = (node / (uint)(32u));
    uint bit_idx = (node % (uint)(32u));
    uint bit_mask = ((uint)(1u) << bit_idx);
    uint flat_idx = ((query * stride) + word_idx);
    uint word = visited[flat_idx];
    uint is_set = (word & bit_mask);
    uint result = (((is_set != (uint)(0u))) ? ((uint)(1u)) : ((uint)(0u)));
    is_visited[tid] = result;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_merge_multi_query_results_f32 = R"kernel(
__kernel void navatala_vector_search_merge_multi_query_results_f32(__global const uint* layer_results, __global const float* layer_distances, __global const uint* n_queries, __global const uint* k, __global const uint* n_layers, __global uint* final_ids, __global float* final_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if ((query_id < nq)) {
    uint flat_idx = ((query_id * k_val) + k_idx);
    uint result_id = layer_results[flat_idx];
    float result_dist = layer_distances[flat_idx];
    final_ids[flat_idx] = result_id;
    final_distances[flat_idx] = result_dist;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_neighbor_distances_f32 = R"kernel(
__kernel void navatala_vector_search_compute_neighbor_distances_f32(__global const float* query, __global const float* dataset, __global const uint* neighbor_ids, __global const uint* n_neighbors, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nn = n_neighbors[0];
  uint d = dim[0];
  if ((tid < nn)) {
    uint neighbor_id = neighbor_ids[tid];
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_greedy_search_f32 = R"kernel(
__kernel void navatala_vector_search_greedy_search_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_point, __global const uint* n_queries, __global const uint* dim, __global const uint* max_degree, __global const uint* search_list_size, __global uint* candidates, __global float* candidate_dists, __global uint* n_candidates) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint ls = search_list_size[0];
  if ((query_id < nq)) {
    float entry_dist = as_float(0x00000000u);
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
    n_candidates[query_id] = (uint)(1u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_robust_prune_f32 = R"kernel(
__kernel void navatala_vector_search_robust_prune_f32(__global const uint* vertex, __global const uint* candidates, __global const float* distances, __global const float* dataset, __global const uint* n_candidates, __global const uint* dim, __global const uint* max_degree, __global const uint* alpha, __global uint* selected, __global uint* n_selected) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  if ((tid == (uint)(0u))) {
    uint limit = (((nc < md)) ? (nc) : (md));
    for (int i = 0; i < (int)(limit); ++i) {
      uint cand = candidates[i];
      selected[i] = cand;
    }
    n_selected[(uint)(0u)] = limit;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_insert_vertex_f32 = R"kernel(
__kernel void navatala_vector_search_insert_vertex_f32(__global const uint* vertex_id, __global const uint* candidates, __global const uint* n_candidates, __global const uint* max_degree, __global uint* graph, __global uint* graph_degrees) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint vid = vertex_id[0];
  uint nc = n_candidates[0];
  uint md = max_degree[0];
  uint n_neighbors = (((nc < md)) ? (nc) : (md));
  if ((tid < n_neighbors)) {
    uint neighbor = candidates[tid];
    uint graph_idx = ((vid * md) + tid);
    graph[graph_idx] = neighbor;
  }
  if ((tid == (uint)(0u))) {
    graph_degrees[vid] = n_neighbors;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_random_init_neighbors_f32 = R"kernel(
__kernel void navatala_vector_search_random_init_neighbors_f32(__global const uint* n_vectors, __global const uint* max_degree, __global const uint* init_degree, __global const uint* seed, __global uint* graph, __global uint* graph_degrees) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint md = max_degree[0];
  uint id = init_degree[0];
  uint s = seed[0];
  if ((vid < nv)) {
    uint rng = (vid + s);
    for (int i = 0; i < (int)(id); ++i) {
      uint old_rng = rng;
      uint temp1 = (old_rng * (uint)(1103515245u));
      uint temp2 = (temp1 + (uint)(12345u));
      uint new_rng = (temp2 & (uint)(2147483647u));
      rng = new_rng;
      uint neighbor_raw = (new_rng % nv);
      uint neighbor = (((neighbor_raw == vid)) ? (((neighbor_raw + (uint)(1u)) % nv)) : (neighbor_raw));
      uint graph_idx = ((vid * md) + i);
      graph[graph_idx] = neighbor;
    }
    graph_degrees[vid] = id;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_extract_search_results_f32 = R"kernel(
__kernel void navatala_vector_search_extract_search_results_f32(__global const uint* candidates, __global const float* candidate_dists, __global const uint* n_candidates, __global const uint* n_queries, __global const uint* k, __global const uint* search_list_size, __global uint* result_ids, __global float* result_dists) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint ls = search_list_size[0];
  uint query_id = (gid / k_val);
  uint k_idx = (gid % k_val);
  if ((query_id < nq)) {
    uint nc = n_candidates[query_id];
    uint src_idx = ((query_id * ls) + k_idx);
    uint dst_idx = ((query_id * k_val) + k_idx);
    if ((k_idx < nc)) {
      uint cand_id = candidates[src_idx];
      float cand_dist = candidate_dists[src_idx];
      result_ids[dst_idx] = cand_id;
      result_dists[dst_idx] = cand_dist;
    } else {
      result_ids[dst_idx] = (uint)(4294967295u);
      result_dists[dst_idx] = as_float(0x7e967699u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_batched_greedy_search_f32 = R"kernel(
__kernel void navatala_vector_search_batched_greedy_search_f32(__global const float* queries, __global const float* dataset, __global const uint* graph, __global const uint* graph_degrees, __global const uint* entry_point, __global const uint* n_queries, __global const uint* n_vectors, __global const uint* dim, __global const uint* max_degree, __global const uint* k, __global uint* result_ids, __global float* result_dists) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_group_id(0))));
  uint tid = ((uint)((int)(get_local_id(0))));
  uint nq = n_queries[0];
  uint d = dim[0];
  uint entry = entry_point[0];
  uint k_val = k[0];
  __local float partial_sums[64];
  if ((query_id < nq)) {
    if ((tid == (uint)(0u))) {
      uint base_idx = (query_id * k_val);
      result_ids[base_idx] = entry;
      result_dists[base_idx] = as_float(0x00000000u);
    }
    if ((tid < k_val)) {
      if ((tid != (uint)(0u))) {
        uint out_idx = ((query_id * k_val) + tid);
        result_ids[out_idx] = (uint)(4294967295u);
        result_dists[out_idx] = as_float(0x7e967699u);
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_centroid_distances_f32 = R"kernel(
__kernel void navatala_vector_search_compute_centroid_distances_f32(__global const float* queries, __global const float* centroids, __global const uint* n_queries, __global const uint* n_lists, __global const uint* dim, __global float* centroid_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_scan_inverted_list_f32 = R"kernel(
__kernel void navatala_vector_search_scan_inverted_list_f32(__global const float* queries, __global const float* list_data, __global const uint* list_indices, __global const uint* list_offsets, __global const uint* probe_indices, __global const uint* n_queries, __global const uint* n_probes, __global const uint* dim, __global const uint* max_list_size, __global float* scan_distances, __global uint* scan_indices, __global uint* scan_counts) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  uint wg_size = (uint)(256u);
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
    uint list_idx_plus1 = (list_idx + (uint)(1u));
    uint list_end = list_offsets[list_idx_plus1];
    uint list_size = (list_end - list_start);
    uint query_base = (query_idx * d);
    uint out_base = (((query_idx * np) * mls) + (probe_idx * mls));
    uint iters = ((list_size / wg_size) + (uint)(1u));
    for (int iter = 0; iter < (int)(iters); ++iter) {
      uint local_vec_idx = ((iter * wg_size) + tid);
      bool valid_vec = (local_vec_idx < list_size);
      if (valid_vec) {
        uint global_vec_idx = (list_start + local_vec_idx);
        uint vec_base = (global_vec_idx * d);
        float dist_acc = as_float(0x00000000u);
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
    bool is_t0 = (tid == (uint)(0u));
    if (is_t0) {
      uint count_offset = ((query_idx * np) + probe_idx);
      scan_counts[count_offset] = list_size;
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_assign_to_cluster_f32 = R"kernel(
__kernel void navatala_vector_search_assign_to_cluster_f32(__global const float* vectors, __global const float* centroids, __global const uint* n_vectors, __global const uint* n_lists, __global const uint* dim, __global uint* labels) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint vec_base = (gid * d);
    float best_dist = as_float(0x7f7fc99eu);
    uint best_label = (uint)(0u);
    for (int c = 0; c < (int)(nl); ++c) {
      uint cent_base = (c * d);
      float dist_acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_initialize_random_graph = R"kernel(
__kernel void navatala_vector_search_initialize_random_graph(__global const uint* n_nodes, __global const uint* graph_degree, __global const uint* seed, __global uint* graph, __global float* graph_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  uint s = seed[0];
  uint total = (nn * deg);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint node_idx = (gid / deg);
    uint slot = (gid % deg);
    uint state0 = ((gid * (uint)(1103515245u)) + (s + (uint)(12345u)));
    uint rand_raw = (state0 % nn);
    bool is_self = (rand_raw == node_idx);
    uint alt = ((node_idx + (uint)(1u)) % nn);
    uint neighbor = ((is_self) ? (alt) : (rand_raw));
    graph[gid] = neighbor;
    graph_distances[gid] = as_float(0x7f7fc99eu);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_graph_distances_f32 = R"kernel(
__kernel void navatala_vector_search_compute_graph_distances_f32(__global const float* dataset, __global const uint* graph, __global const uint* n_nodes, __global const uint* graph_degree, __global const uint* dim, __global float* graph_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    float dist_acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_prune_graph_r_n_g_f32 = R"kernel(
__kernel void navatala_vector_search_prune_graph_r_n_g_f32(__global const float* dataset, __global const uint* graph, __global const float* graph_distances, __global const uint* n_nodes, __global const uint* graph_degree, __global const uint* dim, __global uint* pruned_graph, __global float* pruned_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
      if ((wi != slot)) {
        uint w_slot = (graph_base + wi);
        uint w = graph[w_slot];
        float d_uw = graph_distances[w_slot];
        bool uw_closer = (d_uw < edge_dist);
        if (uw_closer) {
          uint w_base = (w * d);
          float d_wv_acc = as_float(0x00000000u);
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
    uint invalid_marker = (uint)(4294967295u);
    uint out_idx = ((prune) ? (invalid_marker) : (target));
    float out_dist = ((prune) ? (as_float(0x7f7fc99eu)) : (edge_dist));
    pruned_graph[gid] = out_idx;
    pruned_distances[gid] = out_dist;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compact_pruned_graph = R"kernel(
__kernel void navatala_vector_search_compact_pruned_graph(__global const uint* pruned_graph, __global const float* pruned_distances, __global const uint* n_nodes, __global const uint* graph_degree, __global uint* compact_graph, __global float* compact_distances, __global uint* degree_counts) {
  int gid0 = (int)get_global_id(0);
  uint node_idx = ((uint)((int)(get_global_id(0))));
  uint nn = n_nodes[0];
  uint deg = graph_degree[0];
  bool inBounds = (node_idx < nn);
  if (inBounds) {
    uint base = (node_idx * deg);
    uint write_pos = (uint)(0u);
    for (int i = 0; i < (int)(deg); ++i) {
      uint read_idx = (base + i);
      uint neighbor = pruned_graph[read_idx];
      bool is_valid = (neighbor != (uint)(4294967295u));
      if (is_valid) {
        uint wp = write_pos;
        uint write_idx = (base + wp);
        float dist = pruned_distances[read_idx];
        compact_graph[write_idx] = neighbor;
        compact_distances[write_idx] = dist;
        write_pos = (wp + (uint)(1u));
      }
    }
    uint final_count = write_pos;
    degree_counts[node_idx] = final_count;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_vertex_degrees = R"kernel(
__kernel void navatala_vector_search_compute_vertex_degrees(__global const uint* graph, __global const uint* n_vertices, __global const uint* max_degree, __global const uint* invalid_id, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  uint inv = invalid_id[0];
  if ((vid < nv)) {
    uint count = (uint)(0u);
    for (int i = 0; i < (int)(md); ++i) {
      uint idx = ((vid * md) + i);
      uint neighbor = graph[idx];
      if ((neighbor != inv)) {
        uint old_cnt = count;
        count = (old_cnt + (uint)(1u));
      }
    }
    uint final_count = count;
    degrees[vid] = final_count;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_prune_excess_degree = R"kernel(
__kernel void navatala_vector_search_prune_excess_degree(__global uint* graph, __global const float* neighbor_distances, __global uint* degrees, __global const uint* n_vertices, __global const uint* current_max_degree, __global const uint* target_max_degree, __global const uint* invalid_id) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vertices[0];
  uint cmd = current_max_degree[0];
  uint tmd = target_max_degree[0];
  uint inv = invalid_id[0];
  if ((vid < nv)) {
    uint deg = degrees[vid];
    if ((deg > tmd)) {
      for (int i = 0; i < (int)(cmd); ++i) {
        if ((i >= tmd)) {
          uint idx = ((vid * cmd) + i);
          graph[idx] = inv;
        }
      }
      degrees[vid] = tmd;
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compact_graph_edges = R"kernel(
__kernel void navatala_vector_search_compact_graph_edges(__global uint* graph, __global const uint* n_vertices, __global const uint* max_degree, __global const uint* invalid_id, __global uint* degrees) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  uint inv = invalid_id[0];
  if ((vid < nv)) {
    uint write_pos = (uint)(0u);
    for (int i = 0; i < (int)(md); ++i) {
      uint read_idx = ((vid * md) + i);
      uint neighbor = graph[read_idx];
      if ((neighbor != inv)) {
        uint wp = write_pos;
        uint write_idx = ((vid * md) + wp);
        graph[write_idx] = neighbor;
        write_pos = (wp + (uint)(1u));
      }
    }
    uint final_wp = write_pos;
    for (int j = 0; j < (int)(md); ++j) {
      if ((j >= final_wp)) {
        uint fill_idx = ((vid * md) + j);
        graph[fill_idx] = inv;
      }
    }
    degrees[vid] = final_wp;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_validate_graph_integrity = R"kernel(
__kernel void navatala_vector_search_validate_graph_integrity(__global const uint* graph, __global const uint* degrees, __global const uint* n_vertices, __global const uint* max_degree, __global uint* error_flags) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vertices[0];
  uint md = max_degree[0];
  if ((vid < nv)) {
    uint errors = (uint)(0u);
    uint deg = degrees[vid];
    for (int i = 0; i < (int)(deg); ++i) {
      uint idx = ((vid * md) + i);
      uint neighbor = graph[idx];
      if ((neighbor == vid)) {
        uint old_err = errors;
        errors = (old_err | (uint)(1u));
      }
      if ((neighbor >= nv)) {
        uint old_err2 = errors;
        errors = (old_err2 | (uint)(2u));
      }
    }
    uint final_errors = errors;
    error_flags[vid] = final_errors;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_prune_excess_degree_sorted = R"kernel(
__kernel void navatala_vector_search_prune_excess_degree_sorted(__global uint* graph, __global const float* neighbor_distances, __global uint* degrees, __global const uint* n_vertices, __global const uint* current_max_degree, __global const uint* target_max_degree, __global const uint* invalid_id) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_group_id(0))));
  uint tid = ((uint)((int)(get_local_id(0))));
  uint nv = n_vertices[0];
  uint cmd = current_max_degree[0];
  uint tmd = target_max_degree[0];
  uint inv = invalid_id[0];
  __local uint shared_ids[128];
  __local float shared_dists[128];
  if ((vid < nv)) {
    uint deg = degrees[vid];
    if ((deg > tmd)) {
      if ((tid < deg)) {
        uint read_idx = ((vid * cmd) + tid);
        uint neighbor_id = graph[read_idx];
        float neighbor_dist = neighbor_distances[read_idx];
        shared_ids[tid] = neighbor_id;
        shared_dists[tid] = neighbor_dist;
      } else {
        if ((tid < (uint)(128u))) {
          shared_ids[tid] = inv;
          shared_dists[tid] = as_float(0x7e967699u);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int stage = 0; stage < (int)((uint)(7u)); ++stage) {
        for (int substage = 0; substage < (int)((stage + (uint)(1u))); ++substage) {
          uint half_net = ((uint)(1u) << (stage - substage));
          uint partner = (tid ^ half_net);
          uint direction_bit = ((tid >> (stage + (uint)(1u))) & (uint)(1u));
          if ((partner < (uint)(128u))) {
            float my_dist = shared_dists[tid];
            float partner_d = shared_dists[partner];
            uint should_swap = (((tid < partner)) ? ((((direction_bit == (uint)(0u))) ? ((((my_dist > partner_d)) ? ((uint)(1u)) : ((uint)(0u)))) : ((((my_dist < partner_d)) ? ((uint)(1u)) : ((uint)(0u)))))) : ((uint)(0u)));
            if ((should_swap == (uint)(1u))) {
              uint my_id = shared_ids[tid];
              uint partner_id = shared_ids[partner];
              shared_ids[tid] = partner_id;
              shared_ids[partner] = my_id;
              shared_dists[tid] = partner_d;
              shared_dists[partner] = my_dist;
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
      }
      if ((tid < cmd)) {
        uint write_idx = ((vid * cmd) + tid);
        if ((tid < tmd)) {
          uint sorted_id = shared_ids[tid];
          graph[write_idx] = sorted_id;
        } else {
          graph[write_idx] = inv;
        }
      }
      if ((tid == (uint)(0u))) {
        degrees[vid] = tmd;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_normalize_p_q_codebook_f32 = R"kernel(
__kernel void navatala_vector_search_normalize_p_q_codebook_f32(__global const int* codebook_sums_i32, __global const uint* codebook_counts, __global const uint* n_codewords, __global const uint* sub_dim, __global float* codebook) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  uint total = (nc * sd);
  float scale = as_float(0x47800000u);
  if ((tid < total)) {
    uint cw = (tid / sd);
    uint count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > as_float(0x00000000u))) ? ((sum_f32 / count_f32)) : (as_float(0x00000000u)));
    codebook[tid] = _centroid;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_l2_squared_f32 = R"kernel(
__kernel void navatala_vector_search_l2_squared_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_inner_product_f32 = R"kernel(
__kernel void navatala_vector_search_inner_product_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_l1_distance_f32 = R"kernel(
__kernel void navatala_vector_search_l1_distance_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float neg_diff = (as_float(0x00000000u) - diff);
      bool is_neg = (diff < as_float(0x00000000u));
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
const char* k_opencl_navatala_vector_search_batch_l2_to_neighbors_f32 = R"kernel(
__kernel void navatala_vector_search_batch_l2_to_neighbors_f32(__global const float* query, __global const float* dataset, __global const uint* indices, __global const uint* n_neighbors, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = n_neighbors[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint vec_idx = indices[gid];
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_l2_squared_f16 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_vector_search_l2_squared_f16(__global const half* x, __global const half* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_cosine_similarity_f32 = R"kernel(
__kernel void navatala_vector_search_cosine_similarity_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* sim) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float dot = as_float(0x00000000u);
    float norm_x = as_float(0x00000000u);
    float norm_y = as_float(0x00000000u);
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
    float eps = as_float(0x322bcc77u);
    float safe_denom = (denom + eps);
    float cosine = (final_dot / safe_denom);
    uint out_idx = ((gid_x * ny) + gid_y);
    sim[out_idx] = cosine;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_linf_distance_f32 = R"kernel(
__kernel void navatala_vector_search_linf_distance_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float max_diff = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint x_idx = ((gid_x * d) + k);
      uint y_idx = ((gid_y * d) + k);
      float x_val = x[x_idx];
      float y_val = y[y_idx];
      float diff = (x_val - y_val);
      float abs_diff = fabs(diff);
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
const char* k_opencl_navatala_vector_search_hamming_distance_u32 = R"kernel(
__kernel void navatala_vector_search_hamming_distance_u32(__global const uint* x, __global const uint* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim_words, __global uint* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint count = (uint)(0u);
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint xor_val = (x_val ^ y_val);
      uint v0 = xor_val;
      uint m1 = (uint)(1431655765u);
      uint m2 = (uint)(858993459u);
      uint m4 = (uint)(252645135u);
      uint v1 = (v0 - ((v0 >> (uint)(1u)) & m1));
      uint v2 = ((v1 & m2) + ((v1 >> (uint)(2u)) & m2));
      uint v3 = ((v2 + (v2 >> (uint)(4u))) & m4);
      uint v4 = (v3 + (v3 >> (uint)(8u)));
      uint v5 = (v4 + (v4 >> (uint)(16u)));
      uint popcount = (v5 & (uint)(63u));
      uint old_count = count;
      count = (old_count + popcount);
    }
    uint result = count;
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_correlation_distance_f32 = R"kernel(
__kernel void navatala_vector_search_correlation_distance_f32(__global const float* x, __global const float* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float sum_x = as_float(0x00000000u);
    float sum_y = as_float(0x00000000u);
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
    float cov = as_float(0x00000000u);
    float var_x = as_float(0x00000000u);
    float var_y = as_float(0x00000000u);
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
    float eps = as_float(0x322bcc77u);
    float safe_denom = (sqrt_var + eps);
    float correlation = (final_cov / safe_denom);
    float corr_dist = (as_float(0x3f800000u) - correlation);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = corr_dist;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_jaccard_distance_u32 = R"kernel(
__kernel void navatala_vector_search_jaccard_distance_u32(__global const uint* x, __global const uint* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim_words, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint intersection = (uint)(0u);
    uint union_count = (uint)(0u);
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint inter = (x_val & y_val);
      uint uni = (x_val | y_val);
      uint i0 = inter;
      uint m1 = (uint)(1431655765u);
      uint m2 = (uint)(858993459u);
      uint m4 = (uint)(252645135u);
      uint i1 = (i0 - ((i0 >> (uint)(1u)) & m1));
      uint i2 = ((i1 & m2) + ((i1 >> (uint)(2u)) & m2));
      uint i3 = ((i2 + (i2 >> (uint)(4u))) & m4);
      uint i4 = (i3 + (i3 >> (uint)(8u)));
      uint i5 = (i4 + (i4 >> (uint)(16u)));
      uint pop_inter = (i5 & (uint)(63u));
      uint u0 = uni;
      uint u1 = (u0 - ((u0 >> (uint)(1u)) & m1));
      uint u2 = ((u1 & m2) + ((u1 >> (uint)(2u)) & m2));
      uint u3 = ((u2 + (u2 >> (uint)(4u))) & m4);
      uint u4 = (u3 + (u3 >> (uint)(8u)));
      uint u5 = (u4 + (u4 >> (uint)(16u)));
      uint pop_union = (u5 & (uint)(63u));
      uint old_inter = intersection;
      uint old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    uint final_inter = intersection;
    uint final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = as_float(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (as_float(0x3f800000u) - similarity);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_assign_labels_f32 = R"kernel(
__kernel void navatala_vector_search_assign_labels_f32(__global const float* data, __global const float* centroids, __global const uint* n_points, __global const uint* k, __global const uint* dim, __global uint* labels, __global float* min_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = n_points[0];
  uint kVal = k[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float best_dist = as_float(0x7e967699u);
    uint best_label = (uint)(0u);
    for (int c = 0; c < (int)(kVal); ++c) {
      float dist_acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_update_centroids_f32 = R"kernel(
__kernel void navatala_vector_search_update_centroids_f32(__global const float* cluster_sums, __global const uint* cluster_counts, __global const uint* k, __global const uint* dim, __global float* centroids) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    bool is_empty = (count == (uint)(0u));
    float safe_count = ((is_empty) ? (as_float(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_clear_cluster_sums_f32 = R"kernel(
__kernel void navatala_vector_search_clear_cluster_sums_f32(__global const uint* total, __global float* cluster_sums) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = total[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    cluster_sums[gid] = as_float(0x00000000u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_clear_cluster_counts = R"kernel(
__kernel void navatala_vector_search_clear_cluster_counts(__global const uint* k, __global uint* cluster_counts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    cluster_counts[gid] = (uint)(0u);
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_reduce_inertia_f32 = R"kernel(
__kernel void navatala_vector_search_reduce_inertia_f32(__global const float* min_distances, __global const uint* n_points, __global float* partial_sums) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint group_id = ((uint)((int)(get_group_id(0))));
  uint n = n_points[0];
  __local float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (as_float(0x00000000u)));
  sdata[lid] = val;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride = (uint)(128u);
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!((stride > (uint)(0u)))) break;
    uint s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      uint other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    stride = (s / (uint)(2u));
  }
  bool is_thread0 = (lid == (uint)(0u));
  if (is_thread0) {
    float block_sum = sdata[(uint)(0u)];
    partial_sums[group_id] = block_sum;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_bitonic_sort_step_f32 = R"kernel(
__kernel void navatala_vector_search_bitonic_sort_step_f32(__global const uint* n, __global const uint* step, __global const uint* stage, __global float* keys, __global uint* values) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint stepVal = step[0];
  uint stageVal = stage[0];
  uint half_n = (nVal / (uint)(2u));
  bool inBounds = (gid < half_n);
  if (inBounds) {
    uint distance = ((uint)(1u) << (stepVal - stageVal));
    uint block_size = ((uint)(2u) << stepVal);
    uint block_id = (gid / distance);
    uint offset = (gid % distance);
    uint i = ((block_id * ((uint)(2u) * distance)) + offset);
    uint j = (i + distance);
    uint block_start = ((i / block_size) * block_size);
    bool ascending = (((block_start / block_size) % (uint)(2u)) == (uint)(0u));
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
const char* k_opencl_navatala_vector_search_per_row_top_k_f32 = R"kernel(
__kernel void navatala_vector_search_per_row_top_k_f32(__global const float* data, __global const uint* n_rows, __global const uint* row_size, __global const uint* k, __global float* top_values, __global uint* top_indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nr = n_rows[0];
  uint rs = row_size[0];
  uint kVal = k[0];
  uint total = (nr * kVal);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row_idx = (gid / kVal);
    uint k_idx = (gid % kVal);
    uint row_start = (row_idx * rs);
    float best_val = as_float(0x7e967699u);
    uint best_idx = (uint)(0u);
    uint count_smaller = (uint)(0u);
    for (int i = 0; i < (int)(rs); ++i) {
      uint data_idx = (row_start + i);
      float val = data[data_idx];
      uint smaller_count = (uint)(0u);
      for (int j = 0; j < (int)(rs); ++j) {
        uint other_idx = (row_start + j);
        float other_val = data[other_idx];
        bool is_smaller = (other_val < val);
        bool is_equal_earlier = ((other_val == val) && (j < i));
        bool counts = (is_smaller || is_equal_earlier);
        uint old_cnt = smaller_count;
        uint incr = ((counts) ? ((uint)(1u)) : ((uint)(0u)));
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
const char* k_opencl_navatala_vector_search_merge_sorted_f32 = R"kernel(
__kernel void navatala_vector_search_merge_sorted_f32(__global const float* a, __global const float* b, __global const uint* a_indices, __global const uint* b_indices, __global const uint* n_a, __global const uint* n_b, __global float* merged, __global uint* merged_indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint na = n_a[0];
  uint nb = n_b[0];
  uint total = (na + nb);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint lo = (uint)(0u);
    uint hi = gid;
    uint hi_clamped = (((gid > na)) ? (na) : (gid));
    hi = hi_clamped;
    for (int __iter = 0; __iter < 32; ++__iter) {
      if (!((lo < hi))) break;
      uint lo_v = lo;
      uint hi_v = hi;
      uint mid = (lo_v + ((hi_v - lo_v) / (uint)(2u)));
      uint j = (gid - mid);
      float a_val = (((mid < na)) ? (a[mid]) : (as_float(0x7e967699u)));
      float b_prev = (((j > (uint)(0u))) ? (b[(j - (uint)(1u))]) : (as_float(0xfe967699u)));
      bool go_higher = (a_val < b_prev);
      if (go_higher) {
        lo = (mid + (uint)(1u));
      } else {
        hi = mid;
      }
    }
    uint i = lo;
    uint j = (gid - i);
    bool a_available = (i < na);
    bool b_available = (j < nb);
    float a_val = ((a_available) ? (a[i]) : (as_float(0x7e967699u)));
    float b_val = ((b_available) ? (b[j]) : (as_float(0x7e967699u)));
    bool pick_a = (a_val <= b_val);
    float out_val = ((pick_a) ? (a_val) : (b_val));
    uint out_idx = ((pick_a) ? (a_indices[i]) : (b_indices[j]));
    merged[gid] = out_val;
    merged_indices[gid] = out_idx;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_rerank_with_exact_distances_f32 = R"kernel(
__kernel void navatala_vector_search_rerank_with_exact_distances_f32(__global const float* queries, __global const float* dataset, __global const uint* candidate_ids, __global const uint* n_queries, __global const uint* k, __global const uint* dim, __global float* exact_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  uint d = dim[0];
  uint total = (nq * k_val);
  if ((gid < total)) {
    uint query_id = (gid / k_val);
    uint k_idx = (gid % k_val);
    uint cand_id = candidate_ids[gid];
    float dist = as_float(0x00000000u);
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
const char* k_opencl_navatala_vector_search_filter_by_threshold_f32 = R"kernel(
__kernel void navatala_vector_search_filter_by_threshold_f32(__global const float* distances, __global uint* candidate_ids, __global const uint* n_queries, __global const uint* k, __global const float* threshold, __global const uint* invalid_id, __global uint* filtered_counts) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  float thresh = threshold[0];
  uint inv = invalid_id[0];
  if ((query_id < nq)) {
    uint count = (uint)(0u);
    for (int i = 0; i < (int)(k_val); ++i) {
      uint idx = ((query_id * k_val) + i);
      float dist = distances[idx];
      if ((dist > thresh)) {
        candidate_ids[idx] = inv;
      } else {
        uint old_cnt = count;
        count = (old_cnt + (uint)(1u));
      }
    }
    uint final_count = count;
    filtered_counts[query_id] = final_count;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_merge_search_results_f32 = R"kernel(
__kernel void navatala_vector_search_merge_search_results_f32(__global const uint* ids_a, __global const float* dists_a, __global const uint* ids_b, __global const float* dists_b, __global const uint* n_queries, __global const uint* k_a, __global const uint* k_b, __global const uint* k_out, __global uint* merged_ids, __global float* merged_dists) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint ka = k_a[0];
  uint kb = k_b[0];
  uint ko = k_out[0];
  if ((query_id < nq)) {
    uint ptr_a = (uint)(0u);
    uint ptr_b = (uint)(0u);
    uint out_idx = (uint)(0u);
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
      float dist_a = ((a_valid) ? (dist_a_raw) : (as_float(0x7e967699u)));
      float dist_b = ((b_valid) ? (dist_b_raw) : (as_float(0x7e967699u)));
      bool choose_a = (dist_a <= dist_b);
      uint chosen_id = ((choose_a) ? (ids_a[idx_a]) : (ids_b[idx_b]));
      float chosen_dist = ((choose_a) ? (dist_a) : (dist_b));
      uint out_flat = ((query_id * ko) + oi);
      merged_ids[out_flat] = chosen_id;
      merged_dists[out_flat] = chosen_dist;
      if (choose_a) {
        ptr_a = (pa + (uint)(1u));
      } else {
        ptr_b = (pb + (uint)(1u));
      }
      out_idx = (oi + (uint)(1u));
    }
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_recall_f32 = R"kernel(
__kernel void navatala_vector_search_compute_recall_f32(__global const uint* approx_ids, __global const uint* ground_truth_ids, __global const uint* n_queries, __global const uint* k, __global float* recall_per_query) {
  int gid0 = (int)get_global_id(0);
  uint query_id = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint k_val = k[0];
  if ((query_id < nq)) {
    uint hits = (uint)(0u);
    for (int i = 0; i < (int)(k_val); ++i) {
      uint approx_idx = ((query_id * k_val) + i);
      uint approx_id = approx_ids[approx_idx];
      uint found = (uint)(0u);
      for (int j = 0; j < (int)(k_val); ++j) {
        uint gt_idx = ((query_id * k_val) + j);
        uint gt_id = ground_truth_ids[gt_idx];
        if ((approx_id == gt_id)) {
          found = (uint)(1u);
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
const char* k_opencl_navatala_vector_search_scalar_quantize_f32_to_i8 = R"kernel(
__kernel void navatala_vector_search_scalar_quantize_f32_to_i8(__global const float* _input, __global const uint* n_vectors, __global const uint* dim, __global const float* mins, __global const float* maxs, __global char* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if ((gid < total)) {
    uint dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * as_float(0x437f0000u));
    float offset = (scaled - as_float(0x43000000u));
    float neg128 = (as_float(0x00000000u) - as_float(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > as_float(0x42fe0000u))) ? (as_float(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    char quantized = ((char)(quantized_i32));
    _output[gid] = quantized;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32 = R"kernel(
__kernel void navatala_vector_search_scalar_dequantize_i8_to_f32(__global const char* _input, __global const uint* n_vectors, __global const uint* dim, __global const float* mins, __global const float* maxs, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint d = dim[0];
  uint total = (nv * d);
  if ((gid < total)) {
    uint dim_idx = (gid % d);
    char qval = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float qval_f32 = ((float)(qval));
    float offset = (qval_f32 + as_float(0x43000000u));
    float normalized = (offset / as_float(0x437f0000u));
    float range = (max_val - min_val);
    float dequant = ((normalized * range) + min_val);
    _output[gid] = dequant;
  }
}

)kernel";
const char* k_opencl_navatala_vector_search_compute_quantized_distances_i8 = R"kernel(
__kernel void navatala_vector_search_compute_quantized_distances_i8(__global const char* queries, __global const char* dataset, __global const uint* n_queries, __global const uint* n_dataset, __global const uint* dim, __global const float* scale, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
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
const char* k_opencl_navatala_vector_search_encode_p_q_vectors_f32 = R"kernel(
__kernel void navatala_vector_search_encode_p_q_vectors_f32(__global const float* subvectors, __global const float* codebook, __global const uint* n_vectors, __global const uint* n_codewords, __global const uint* sub_dim, __global uchar* codes) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  if ((vid < nv)) {
    uint best_cw = (uint)(0u);
    float best_dist = as_float(0x7e967699u);
    for (int cw = 0; cw < (int)(nc); ++cw) {
      float dist = as_float(0x00000000u);
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
      if ((cw_dist < curr_best)) {
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
const char* k_opencl_navatala_vector_search_decode_p_q_vectors_f32 = R"kernel(
__kernel void navatala_vector_search_decode_p_q_vectors_f32(__global const uchar* codes, __global const float* codebook, __global const uint* n_vectors, __global const uint* sub_dim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  uint total = (nv * sd);
  if ((gid < total)) {
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

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_init_seeds_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_init_seeds_f32 = {
  1,
  "navatala_vector_search_init_seeds_f32",
  "opencl",
  "navatala_vector_search_init_seeds_f32",
  "kernel:opencl:navatala_vector_search_init_seeds_f32",
  "abi-r1:opencl:navatala_vector_search_init_seeds_f32",
  "abi-r1:opencl:navatala_vector_search_init_seeds_f32",
  9,
  kAbiArgs_opencl_navatala_vector_search_init_seeds_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_expand_neighbors_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_expand_neighbors_f32 = {
  1,
  "navatala_vector_search_expand_neighbors_f32",
  "opencl",
  "navatala_vector_search_expand_neighbors_f32",
  "kernel:opencl:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_expand_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_expand_neighbors_f32",
  10,
  kAbiArgs_opencl_navatala_vector_search_expand_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_clear_visited[] = {
  { "hash_size", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_clear_visited = {
  1,
  "navatala_vector_search_clear_visited",
  "opencl",
  "navatala_vector_search_clear_visited",
  "kernel:opencl:navatala_vector_search_clear_visited",
  "abi-r1:opencl:navatala_vector_search_clear_visited",
  "abi-r1:opencl:navatala_vector_search_clear_visited",
  2,
  kAbiArgs_opencl_navatala_vector_search_clear_visited
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_extract_results_f32[] = {
  { "sorted_candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sorted_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "itopk_size", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_extract_results_f32 = {
  1,
  "navatala_vector_search_extract_results_f32",
  "opencl",
  "navatala_vector_search_extract_results_f32",
  "kernel:opencl:navatala_vector_search_extract_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_results_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_extract_results_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_distances_batch_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_distances_batch_f32 = {
  1,
  "navatala_vector_search_compute_distances_batch_f32",
  "opencl",
  "navatala_vector_search_compute_distances_batch_f32",
  "kernel:opencl:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:opencl:navatala_vector_search_compute_distances_batch_f32",
  "abi-r1:opencl:navatala_vector_search_compute_distances_batch_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_compute_distances_batch_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_search_layer_greedy_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_search_layer_greedy_f32 = {
  1,
  "navatala_vector_search_search_layer_greedy_f32",
  "opencl",
  "navatala_vector_search_search_layer_greedy_f32",
  "kernel:opencl:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:opencl:navatala_vector_search_search_layer_greedy_f32",
  "abi-r1:opencl:navatala_vector_search_search_layer_greedy_f32",
  12,
  kAbiArgs_opencl_navatala_vector_search_search_layer_greedy_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_select_neighbors_simple_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "m", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "selected", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "selected_count", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_select_neighbors_simple_f32 = {
  1,
  "navatala_vector_search_select_neighbors_simple_f32",
  "opencl",
  "navatala_vector_search_select_neighbors_simple_f32",
  "kernel:opencl:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:opencl:navatala_vector_search_select_neighbors_simple_f32",
  "abi-r1:opencl:navatala_vector_search_select_neighbors_simple_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_select_neighbors_simple_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_select_neighbors_heuristic_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_select_neighbors_heuristic_f32 = {
  1,
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "opencl",
  "navatala_vector_search_select_neighbors_heuristic_f32",
  "kernel:opencl:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:opencl:navatala_vector_search_select_neighbors_heuristic_f32",
  "abi-r1:opencl:navatala_vector_search_select_neighbors_heuristic_f32",
  9,
  kAbiArgs_opencl_navatala_vector_search_select_neighbors_heuristic_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_update_candidate_list_f32[] = {
  { "candidates", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "new_distances", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_current", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_new", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ef", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_merged", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_update_candidate_list_f32 = {
  1,
  "navatala_vector_search_update_candidate_list_f32",
  "opencl",
  "navatala_vector_search_update_candidate_list_f32",
  "kernel:opencl:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:opencl:navatala_vector_search_update_candidate_list_f32",
  "abi-r1:opencl:navatala_vector_search_update_candidate_list_f32",
  8,
  kAbiArgs_opencl_navatala_vector_search_update_candidate_list_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_extract_layer_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_found", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_extract_layer_results_f32 = {
  1,
  "navatala_vector_search_extract_layer_results_f32",
  "opencl",
  "navatala_vector_search_extract_layer_results_f32",
  "kernel:opencl:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_layer_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_layer_results_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_extract_layer_results_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_init_search_state_f32[] = {
  { "global_entry", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_queries", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "entry_points", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_init_search_state_f32 = {
  1,
  "navatala_vector_search_init_search_state_f32",
  "opencl",
  "navatala_vector_search_init_search_state_f32",
  "kernel:opencl:navatala_vector_search_init_search_state_f32",
  "abi-r1:opencl:navatala_vector_search_init_search_state_f32",
  "abi-r1:opencl:navatala_vector_search_init_search_state_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_init_search_state_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_mark_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_mark_visited_batch = {
  1,
  "navatala_vector_search_mark_visited_batch",
  "opencl",
  "navatala_vector_search_mark_visited_batch",
  "kernel:opencl:navatala_vector_search_mark_visited_batch",
  "abi-r1:opencl:navatala_vector_search_mark_visited_batch",
  "abi-r1:opencl:navatala_vector_search_mark_visited_batch",
  5,
  kAbiArgs_opencl_navatala_vector_search_mark_visited_batch
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_check_visited_batch[] = {
  { "node_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "query_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited_stride", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "visited", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "is_visited", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_check_visited_batch = {
  1,
  "navatala_vector_search_check_visited_batch",
  "opencl",
  "navatala_vector_search_check_visited_batch",
  "kernel:opencl:navatala_vector_search_check_visited_batch",
  "abi-r1:opencl:navatala_vector_search_check_visited_batch",
  "abi-r1:opencl:navatala_vector_search_check_visited_batch",
  6,
  kAbiArgs_opencl_navatala_vector_search_check_visited_batch
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_merge_multi_query_results_f32[] = {
  { "layer_results", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "layer_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_layers", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "final_ids", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "final_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_merge_multi_query_results_f32 = {
  1,
  "navatala_vector_search_merge_multi_query_results_f32",
  "opencl",
  "navatala_vector_search_merge_multi_query_results_f32",
  "kernel:opencl:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:opencl:navatala_vector_search_merge_multi_query_results_f32",
  "abi-r1:opencl:navatala_vector_search_merge_multi_query_results_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_merge_multi_query_results_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_neighbor_distances_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_neighbor_distances_f32 = {
  1,
  "navatala_vector_search_compute_neighbor_distances_f32",
  "opencl",
  "navatala_vector_search_compute_neighbor_distances_f32",
  "kernel:opencl:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_neighbor_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_neighbor_distances_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_compute_neighbor_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_greedy_search_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_greedy_search_f32 = {
  1,
  "navatala_vector_search_greedy_search_f32",
  "opencl",
  "navatala_vector_search_greedy_search_f32",
  "kernel:opencl:navatala_vector_search_greedy_search_f32",
  "abi-r1:opencl:navatala_vector_search_greedy_search_f32",
  "abi-r1:opencl:navatala_vector_search_greedy_search_f32",
  12,
  kAbiArgs_opencl_navatala_vector_search_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_robust_prune_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_robust_prune_f32 = {
  1,
  "navatala_vector_search_robust_prune_f32",
  "opencl",
  "navatala_vector_search_robust_prune_f32",
  "kernel:opencl:navatala_vector_search_robust_prune_f32",
  "abi-r1:opencl:navatala_vector_search_robust_prune_f32",
  "abi-r1:opencl:navatala_vector_search_robust_prune_f32",
  10,
  kAbiArgs_opencl_navatala_vector_search_robust_prune_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_insert_vertex_f32[] = {
  { "vertex_id", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "candidates", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_insert_vertex_f32 = {
  1,
  "navatala_vector_search_insert_vertex_f32",
  "opencl",
  "navatala_vector_search_insert_vertex_f32",
  "kernel:opencl:navatala_vector_search_insert_vertex_f32",
  "abi-r1:opencl:navatala_vector_search_insert_vertex_f32",
  "abi-r1:opencl:navatala_vector_search_insert_vertex_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_insert_vertex_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_random_init_neighbors_f32[] = {
  { "n_vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "init_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_degrees", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_random_init_neighbors_f32 = {
  1,
  "navatala_vector_search_random_init_neighbors_f32",
  "opencl",
  "navatala_vector_search_random_init_neighbors_f32",
  "kernel:opencl:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_random_init_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_random_init_neighbors_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_random_init_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_extract_search_results_f32[] = {
  { "candidates", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_dists", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_candidates", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "search_list_size", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result_ids", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "result_dists", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_extract_search_results_f32 = {
  1,
  "navatala_vector_search_extract_search_results_f32",
  "opencl",
  "navatala_vector_search_extract_search_results_f32",
  "kernel:opencl:navatala_vector_search_extract_search_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_search_results_f32",
  "abi-r1:opencl:navatala_vector_search_extract_search_results_f32",
  8,
  kAbiArgs_opencl_navatala_vector_search_extract_search_results_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_batched_greedy_search_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_batched_greedy_search_f32 = {
  1,
  "navatala_vector_search_batched_greedy_search_f32",
  "opencl",
  "navatala_vector_search_batched_greedy_search_f32",
  "kernel:opencl:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:opencl:navatala_vector_search_batched_greedy_search_f32",
  "abi-r1:opencl:navatala_vector_search_batched_greedy_search_f32",
  12,
  kAbiArgs_opencl_navatala_vector_search_batched_greedy_search_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_centroid_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroid_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_centroid_distances_f32 = {
  1,
  "navatala_vector_search_compute_centroid_distances_f32",
  "opencl",
  "navatala_vector_search_compute_centroid_distances_f32",
  "kernel:opencl:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_centroid_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_centroid_distances_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_compute_centroid_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_scan_inverted_list_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_scan_inverted_list_f32 = {
  1,
  "navatala_vector_search_scan_inverted_list_f32",
  "opencl",
  "navatala_vector_search_scan_inverted_list_f32",
  "kernel:opencl:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:opencl:navatala_vector_search_scan_inverted_list_f32",
  "abi-r1:opencl:navatala_vector_search_scan_inverted_list_f32",
  12,
  kAbiArgs_opencl_navatala_vector_search_scan_inverted_list_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_assign_to_cluster_f32[] = {
  { "vectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_lists", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_assign_to_cluster_f32 = {
  1,
  "navatala_vector_search_assign_to_cluster_f32",
  "opencl",
  "navatala_vector_search_assign_to_cluster_f32",
  "kernel:opencl:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:opencl:navatala_vector_search_assign_to_cluster_f32",
  "abi-r1:opencl:navatala_vector_search_assign_to_cluster_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_assign_to_cluster_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_initialize_random_graph[] = {
  { "n_nodes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "seed", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_initialize_random_graph = {
  1,
  "navatala_vector_search_initialize_random_graph",
  "opencl",
  "navatala_vector_search_initialize_random_graph",
  "kernel:opencl:navatala_vector_search_initialize_random_graph",
  "abi-r1:opencl:navatala_vector_search_initialize_random_graph",
  "abi-r1:opencl:navatala_vector_search_initialize_random_graph",
  5,
  kAbiArgs_opencl_navatala_vector_search_initialize_random_graph
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_graph_distances_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_graph_distances_f32 = {
  1,
  "navatala_vector_search_compute_graph_distances_f32",
  "opencl",
  "navatala_vector_search_compute_graph_distances_f32",
  "kernel:opencl:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_graph_distances_f32",
  "abi-r1:opencl:navatala_vector_search_compute_graph_distances_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_compute_graph_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_prune_graph_r_n_g_f32[] = {
  { "dataset", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "graph_distances", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "pruned_graph", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_prune_graph_r_n_g_f32 = {
  1,
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "opencl",
  "navatala_vector_search_prune_graph_r_n_g_f32",
  "kernel:opencl:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:opencl:navatala_vector_search_prune_graph_r_n_g_f32",
  "abi-r1:opencl:navatala_vector_search_prune_graph_r_n_g_f32",
  8,
  kAbiArgs_opencl_navatala_vector_search_prune_graph_r_n_g_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compact_pruned_graph[] = {
  { "pruned_graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pruned_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_nodes", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "graph_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "compact_graph", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "compact_distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degree_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compact_pruned_graph = {
  1,
  "navatala_vector_search_compact_pruned_graph",
  "opencl",
  "navatala_vector_search_compact_pruned_graph",
  "kernel:opencl:navatala_vector_search_compact_pruned_graph",
  "abi-r1:opencl:navatala_vector_search_compact_pruned_graph",
  "abi-r1:opencl:navatala_vector_search_compact_pruned_graph",
  7,
  kAbiArgs_opencl_navatala_vector_search_compact_pruned_graph
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_vertex_degrees[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_vertex_degrees = {
  1,
  "navatala_vector_search_compute_vertex_degrees",
  "opencl",
  "navatala_vector_search_compute_vertex_degrees",
  "kernel:opencl:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:opencl:navatala_vector_search_compute_vertex_degrees",
  "abi-r1:opencl:navatala_vector_search_compute_vertex_degrees",
  5,
  kAbiArgs_opencl_navatala_vector_search_compute_vertex_degrees
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_prune_excess_degree[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_prune_excess_degree = {
  1,
  "navatala_vector_search_prune_excess_degree",
  "opencl",
  "navatala_vector_search_prune_excess_degree",
  "kernel:opencl:navatala_vector_search_prune_excess_degree",
  "abi-r1:opencl:navatala_vector_search_prune_excess_degree",
  "abi-r1:opencl:navatala_vector_search_prune_excess_degree",
  7,
  kAbiArgs_opencl_navatala_vector_search_prune_excess_degree
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compact_graph_edges[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "degrees", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compact_graph_edges = {
  1,
  "navatala_vector_search_compact_graph_edges",
  "opencl",
  "navatala_vector_search_compact_graph_edges",
  "kernel:opencl:navatala_vector_search_compact_graph_edges",
  "abi-r1:opencl:navatala_vector_search_compact_graph_edges",
  "abi-r1:opencl:navatala_vector_search_compact_graph_edges",
  5,
  kAbiArgs_opencl_navatala_vector_search_compact_graph_edges
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_validate_graph_integrity[] = {
  { "graph", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "max_degree", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "error_flags", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_validate_graph_integrity = {
  1,
  "navatala_vector_search_validate_graph_integrity",
  "opencl",
  "navatala_vector_search_validate_graph_integrity",
  "kernel:opencl:navatala_vector_search_validate_graph_integrity",
  "abi-r1:opencl:navatala_vector_search_validate_graph_integrity",
  "abi-r1:opencl:navatala_vector_search_validate_graph_integrity",
  5,
  kAbiArgs_opencl_navatala_vector_search_validate_graph_integrity
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_prune_excess_degree_sorted[] = {
  { "graph", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbor_distances", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "degrees", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vertices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "current_max_degree", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "target_max_degree", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_prune_excess_degree_sorted = {
  1,
  "navatala_vector_search_prune_excess_degree_sorted",
  "opencl",
  "navatala_vector_search_prune_excess_degree_sorted",
  "kernel:opencl:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:opencl:navatala_vector_search_prune_excess_degree_sorted",
  "abi-r1:opencl:navatala_vector_search_prune_excess_degree_sorted",
  7,
  kAbiArgs_opencl_navatala_vector_search_prune_excess_degree_sorted
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_normalize_p_q_codebook_f32[] = {
  { "codebook_sums_i32", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_codewords", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codebook", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_normalize_p_q_codebook_f32 = {
  1,
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "opencl",
  "navatala_vector_search_normalize_p_q_codebook_f32",
  "kernel:opencl:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:opencl:navatala_vector_search_normalize_p_q_codebook_f32",
  "abi-r1:opencl:navatala_vector_search_normalize_p_q_codebook_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_normalize_p_q_codebook_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_l2_squared_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_l2_squared_f32 = {
  1,
  "navatala_vector_search_l2_squared_f32",
  "opencl",
  "navatala_vector_search_l2_squared_f32",
  "kernel:opencl:navatala_vector_search_l2_squared_f32",
  "abi-r1:opencl:navatala_vector_search_l2_squared_f32",
  "abi-r1:opencl:navatala_vector_search_l2_squared_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_l2_squared_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_inner_product_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_inner_product_f32 = {
  1,
  "navatala_vector_search_inner_product_f32",
  "opencl",
  "navatala_vector_search_inner_product_f32",
  "kernel:opencl:navatala_vector_search_inner_product_f32",
  "abi-r1:opencl:navatala_vector_search_inner_product_f32",
  "abi-r1:opencl:navatala_vector_search_inner_product_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_l1_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_l1_distance_f32 = {
  1,
  "navatala_vector_search_l1_distance_f32",
  "opencl",
  "navatala_vector_search_l1_distance_f32",
  "kernel:opencl:navatala_vector_search_l1_distance_f32",
  "abi-r1:opencl:navatala_vector_search_l1_distance_f32",
  "abi-r1:opencl:navatala_vector_search_l1_distance_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_l1_distance_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_batch_l2_to_neighbors_f32[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_neighbors", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_batch_l2_to_neighbors_f32 = {
  1,
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "opencl",
  "navatala_vector_search_batch_l2_to_neighbors_f32",
  "kernel:opencl:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_batch_l2_to_neighbors_f32",
  "abi-r1:opencl:navatala_vector_search_batch_l2_to_neighbors_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_batch_l2_to_neighbors_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_l2_squared_f16[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_l2_squared_f16 = {
  1,
  "navatala_vector_search_l2_squared_f16",
  "opencl",
  "navatala_vector_search_l2_squared_f16",
  "kernel:opencl:navatala_vector_search_l2_squared_f16",
  "abi-r1:opencl:navatala_vector_search_l2_squared_f16",
  "abi-r1:opencl:navatala_vector_search_l2_squared_f16",
  6,
  kAbiArgs_opencl_navatala_vector_search_l2_squared_f16
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_cosine_similarity_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sim", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_cosine_similarity_f32 = {
  1,
  "navatala_vector_search_cosine_similarity_f32",
  "opencl",
  "navatala_vector_search_cosine_similarity_f32",
  "kernel:opencl:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:opencl:navatala_vector_search_cosine_similarity_f32",
  "abi-r1:opencl:navatala_vector_search_cosine_similarity_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_cosine_similarity_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_linf_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_linf_distance_f32 = {
  1,
  "navatala_vector_search_linf_distance_f32",
  "opencl",
  "navatala_vector_search_linf_distance_f32",
  "kernel:opencl:navatala_vector_search_linf_distance_f32",
  "abi-r1:opencl:navatala_vector_search_linf_distance_f32",
  "abi-r1:opencl:navatala_vector_search_linf_distance_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_linf_distance_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_hamming_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_hamming_distance_u32 = {
  1,
  "navatala_vector_search_hamming_distance_u32",
  "opencl",
  "navatala_vector_search_hamming_distance_u32",
  "kernel:opencl:navatala_vector_search_hamming_distance_u32",
  "abi-r1:opencl:navatala_vector_search_hamming_distance_u32",
  "abi-r1:opencl:navatala_vector_search_hamming_distance_u32",
  6,
  kAbiArgs_opencl_navatala_vector_search_hamming_distance_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_correlation_distance_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_correlation_distance_f32 = {
  1,
  "navatala_vector_search_correlation_distance_f32",
  "opencl",
  "navatala_vector_search_correlation_distance_f32",
  "kernel:opencl:navatala_vector_search_correlation_distance_f32",
  "abi-r1:opencl:navatala_vector_search_correlation_distance_f32",
  "abi-r1:opencl:navatala_vector_search_correlation_distance_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_correlation_distance_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_jaccard_distance_u32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_y", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim_words", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dist", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_jaccard_distance_u32 = {
  1,
  "navatala_vector_search_jaccard_distance_u32",
  "opencl",
  "navatala_vector_search_jaccard_distance_u32",
  "kernel:opencl:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:opencl:navatala_vector_search_jaccard_distance_u32",
  "abi-r1:opencl:navatala_vector_search_jaccard_distance_u32",
  6,
  kAbiArgs_opencl_navatala_vector_search_jaccard_distance_u32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_assign_labels_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "centroids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "labels", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "min_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_assign_labels_f32 = {
  1,
  "navatala_vector_search_assign_labels_f32",
  "opencl",
  "navatala_vector_search_assign_labels_f32",
  "kernel:opencl:navatala_vector_search_assign_labels_f32",
  "abi-r1:opencl:navatala_vector_search_assign_labels_f32",
  "abi-r1:opencl:navatala_vector_search_assign_labels_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_assign_labels_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_update_centroids_f32[] = {
  { "cluster_sums", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "centroids", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_update_centroids_f32 = {
  1,
  "navatala_vector_search_update_centroids_f32",
  "opencl",
  "navatala_vector_search_update_centroids_f32",
  "kernel:opencl:navatala_vector_search_update_centroids_f32",
  "abi-r1:opencl:navatala_vector_search_update_centroids_f32",
  "abi-r1:opencl:navatala_vector_search_update_centroids_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_update_centroids_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_clear_cluster_sums_f32[] = {
  { "total", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_sums", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_clear_cluster_sums_f32 = {
  1,
  "navatala_vector_search_clear_cluster_sums_f32",
  "opencl",
  "navatala_vector_search_clear_cluster_sums_f32",
  "kernel:opencl:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:opencl:navatala_vector_search_clear_cluster_sums_f32",
  "abi-r1:opencl:navatala_vector_search_clear_cluster_sums_f32",
  2,
  kAbiArgs_opencl_navatala_vector_search_clear_cluster_sums_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_clear_cluster_counts[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "cluster_counts", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_clear_cluster_counts = {
  1,
  "navatala_vector_search_clear_cluster_counts",
  "opencl",
  "navatala_vector_search_clear_cluster_counts",
  "kernel:opencl:navatala_vector_search_clear_cluster_counts",
  "abi-r1:opencl:navatala_vector_search_clear_cluster_counts",
  "abi-r1:opencl:navatala_vector_search_clear_cluster_counts",
  2,
  kAbiArgs_opencl_navatala_vector_search_clear_cluster_counts
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_reduce_inertia_f32[] = {
  { "min_distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_points", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partial_sums", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_reduce_inertia_f32 = {
  1,
  "navatala_vector_search_reduce_inertia_f32",
  "opencl",
  "navatala_vector_search_reduce_inertia_f32",
  "kernel:opencl:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:opencl:navatala_vector_search_reduce_inertia_f32",
  "abi-r1:opencl:navatala_vector_search_reduce_inertia_f32",
  3,
  kAbiArgs_opencl_navatala_vector_search_reduce_inertia_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_bitonic_sort_step_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "step", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stage", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_bitonic_sort_step_f32 = {
  1,
  "navatala_vector_search_bitonic_sort_step_f32",
  "opencl",
  "navatala_vector_search_bitonic_sort_step_f32",
  "kernel:opencl:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:opencl:navatala_vector_search_bitonic_sort_step_f32",
  "abi-r1:opencl:navatala_vector_search_bitonic_sort_step_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_bitonic_sort_step_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_per_row_top_k_f32[] = {
  { "data", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_rows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "row_size", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "top_values", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "top_indices", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_per_row_top_k_f32 = {
  1,
  "navatala_vector_search_per_row_top_k_f32",
  "opencl",
  "navatala_vector_search_per_row_top_k_f32",
  "kernel:opencl:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:opencl:navatala_vector_search_per_row_top_k_f32",
  "abi-r1:opencl:navatala_vector_search_per_row_top_k_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_per_row_top_k_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_merge_sorted_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "a_indices", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b_indices", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_a", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_b", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "merged", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "merged_indices", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_merge_sorted_f32 = {
  1,
  "navatala_vector_search_merge_sorted_f32",
  "opencl",
  "navatala_vector_search_merge_sorted_f32",
  "kernel:opencl:navatala_vector_search_merge_sorted_f32",
  "abi-r1:opencl:navatala_vector_search_merge_sorted_f32",
  "abi-r1:opencl:navatala_vector_search_merge_sorted_f32",
  8,
  kAbiArgs_opencl_navatala_vector_search_merge_sorted_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_rerank_with_exact_distances_f32[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "exact_distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_rerank_with_exact_distances_f32 = {
  1,
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "opencl",
  "navatala_vector_search_rerank_with_exact_distances_f32",
  "kernel:opencl:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:opencl:navatala_vector_search_rerank_with_exact_distances_f32",
  "abi-r1:opencl:navatala_vector_search_rerank_with_exact_distances_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_rerank_with_exact_distances_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_filter_by_threshold_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "candidate_ids", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "threshold", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "invalid_id", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "filtered_counts", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_filter_by_threshold_f32 = {
  1,
  "navatala_vector_search_filter_by_threshold_f32",
  "opencl",
  "navatala_vector_search_filter_by_threshold_f32",
  "kernel:opencl:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:opencl:navatala_vector_search_filter_by_threshold_f32",
  "abi-r1:opencl:navatala_vector_search_filter_by_threshold_f32",
  7,
  kAbiArgs_opencl_navatala_vector_search_filter_by_threshold_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_merge_search_results_f32[] = {
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
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_merge_search_results_f32 = {
  1,
  "navatala_vector_search_merge_search_results_f32",
  "opencl",
  "navatala_vector_search_merge_search_results_f32",
  "kernel:opencl:navatala_vector_search_merge_search_results_f32",
  "abi-r1:opencl:navatala_vector_search_merge_search_results_f32",
  "abi-r1:opencl:navatala_vector_search_merge_search_results_f32",
  10,
  kAbiArgs_opencl_navatala_vector_search_merge_search_results_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_recall_f32[] = {
  { "approx_ids", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ground_truth_ids", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recall_per_query", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_recall_f32 = {
  1,
  "navatala_vector_search_compute_recall_f32",
  "opencl",
  "navatala_vector_search_compute_recall_f32",
  "kernel:opencl:navatala_vector_search_compute_recall_f32",
  "abi-r1:opencl:navatala_vector_search_compute_recall_f32",
  "abi-r1:opencl:navatala_vector_search_compute_recall_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_compute_recall_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_scalar_quantize_f32_to_i8[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_scalar_quantize_f32_to_i8 = {
  1,
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "opencl",
  "navatala_vector_search_scalar_quantize_f32_to_i8",
  "kernel:opencl:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:opencl:navatala_vector_search_scalar_quantize_f32_to_i8",
  "abi-r1:opencl:navatala_vector_search_scalar_quantize_f32_to_i8",
  6,
  kAbiArgs_opencl_navatala_vector_search_scalar_quantize_f32_to_i8
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "mins", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "maxs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "output", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32 = {
  1,
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "opencl",
  "navatala_vector_search_scalar_dequantize_i8_to_f32",
  "kernel:opencl:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:opencl:navatala_vector_search_scalar_dequantize_i8_to_f32",
  "abi-r1:opencl:navatala_vector_search_scalar_dequantize_i8_to_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_compute_quantized_distances_i8[] = {
  { "queries", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dataset", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_queries", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_dataset", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scale", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_compute_quantized_distances_i8 = {
  1,
  "navatala_vector_search_compute_quantized_distances_i8",
  "opencl",
  "navatala_vector_search_compute_quantized_distances_i8",
  "kernel:opencl:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:opencl:navatala_vector_search_compute_quantized_distances_i8",
  "abi-r1:opencl:navatala_vector_search_compute_quantized_distances_i8",
  7,
  kAbiArgs_opencl_navatala_vector_search_compute_quantized_distances_i8
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_encode_p_q_vectors_f32[] = {
  { "subvectors", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "n_codewords", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "codes", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_encode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_encode_p_q_vectors_f32",
  "opencl",
  "navatala_vector_search_encode_p_q_vectors_f32",
  "kernel:opencl:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:opencl:navatala_vector_search_encode_p_q_vectors_f32",
  "abi-r1:opencl:navatala_vector_search_encode_p_q_vectors_f32",
  6,
  kAbiArgs_opencl_navatala_vector_search_encode_p_q_vectors_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_vector_search_decode_p_q_vectors_f32[] = {
  { "codes", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "codebook", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n_vectors", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sub_dim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_vector_search_decode_p_q_vectors_f32 = {
  1,
  "navatala_vector_search_decode_p_q_vectors_f32",
  "opencl",
  "navatala_vector_search_decode_p_q_vectors_f32",
  "kernel:opencl:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:opencl:navatala_vector_search_decode_p_q_vectors_f32",
  "abi-r1:opencl:navatala_vector_search_decode_p_q_vectors_f32",
  5,
  kAbiArgs_opencl_navatala_vector_search_decode_p_q_vectors_f32
};

bool tryGetKernelAbiManifest_opencl_vector_search(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_init_seeds_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_expand_neighbors_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_visited") {
    out = &kAbiManifest_opencl_navatala_vector_search_clear_visited;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_results_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_extract_results_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_distances_batch_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_search_layer_greedy_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_select_neighbors_simple_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_select_neighbors_heuristic_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_update_candidate_list_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_extract_layer_results_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_init_search_state_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out = &kAbiManifest_opencl_navatala_vector_search_mark_visited_batch;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_check_visited_batch") {
    out = &kAbiManifest_opencl_navatala_vector_search_check_visited_batch;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_merge_multi_query_results_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_neighbor_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_greedy_search_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_robust_prune_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_insert_vertex_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_random_init_neighbors_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_extract_search_results_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_batched_greedy_search_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_centroid_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_scan_inverted_list_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_assign_to_cluster_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out = &kAbiManifest_opencl_navatala_vector_search_initialize_random_graph;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_graph_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_prune_graph_r_n_g_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out = &kAbiManifest_opencl_navatala_vector_search_compact_pruned_graph;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_vertex_degrees;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out = &kAbiManifest_opencl_navatala_vector_search_prune_excess_degree;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out = &kAbiManifest_opencl_navatala_vector_search_compact_graph_edges;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out = &kAbiManifest_opencl_navatala_vector_search_validate_graph_integrity;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out = &kAbiManifest_opencl_navatala_vector_search_prune_excess_degree_sorted;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_normalize_p_q_codebook_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_l2_squared_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_inner_product_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_inner_product_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_l1_distance_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_batch_l2_to_neighbors_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out = &kAbiManifest_opencl_navatala_vector_search_l2_squared_f16;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_cosine_similarity_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_linf_distance_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out = &kAbiManifest_opencl_navatala_vector_search_hamming_distance_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_correlation_distance_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out = &kAbiManifest_opencl_navatala_vector_search_jaccard_distance_u32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_assign_labels_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_update_centroids_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_clear_cluster_sums_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out = &kAbiManifest_opencl_navatala_vector_search_clear_cluster_counts;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_reduce_inertia_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_bitonic_sort_step_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_per_row_top_k_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_merge_sorted_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_rerank_with_exact_distances_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_filter_by_threshold_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_merge_search_results_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_recall_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out = &kAbiManifest_opencl_navatala_vector_search_scalar_quantize_f32_to_i8;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out = &kAbiManifest_opencl_navatala_vector_search_compute_quantized_distances_i8;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_encode_p_q_vectors_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out = &kAbiManifest_opencl_navatala_vector_search_decode_p_q_vectors_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_vector_search(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_vector_search_init_seeds_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_init_seeds_f32";
    std::string_view sv(k_opencl_navatala_vector_search_init_seeds_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_expand_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_expand_neighbors_f32";
    std::string_view sv(k_opencl_navatala_vector_search_expand_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_visited") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_clear_visited";
    std::string_view sv(k_opencl_navatala_vector_search_clear_visited);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_extract_results_f32";
    std::string_view sv(k_opencl_navatala_vector_search_extract_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_distances_batch_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_distances_batch_f32";
    std::string_view sv(k_opencl_navatala_vector_search_compute_distances_batch_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_search_layer_greedy_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_search_layer_greedy_f32";
    std::string_view sv(k_opencl_navatala_vector_search_search_layer_greedy_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_select_neighbors_simple_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_select_neighbors_simple_f32";
    std::string_view sv(k_opencl_navatala_vector_search_select_neighbors_simple_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_select_neighbors_heuristic_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_select_neighbors_heuristic_f32";
    std::string_view sv(k_opencl_navatala_vector_search_select_neighbors_heuristic_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_update_candidate_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_update_candidate_list_f32";
    std::string_view sv(k_opencl_navatala_vector_search_update_candidate_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_layer_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_extract_layer_results_f32";
    std::string_view sv(k_opencl_navatala_vector_search_extract_layer_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_init_search_state_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_init_search_state_f32";
    std::string_view sv(k_opencl_navatala_vector_search_init_search_state_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_mark_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_mark_visited_batch";
    std::string_view sv(k_opencl_navatala_vector_search_mark_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_check_visited_batch") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_check_visited_batch";
    std::string_view sv(k_opencl_navatala_vector_search_check_visited_batch);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_multi_query_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_merge_multi_query_results_f32";
    std::string_view sv(k_opencl_navatala_vector_search_merge_multi_query_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_neighbor_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_neighbor_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_compute_neighbor_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_greedy_search_f32";
    std::string_view sv(k_opencl_navatala_vector_search_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_robust_prune_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_robust_prune_f32";
    std::string_view sv(k_opencl_navatala_vector_search_robust_prune_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_insert_vertex_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_insert_vertex_f32";
    std::string_view sv(k_opencl_navatala_vector_search_insert_vertex_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_random_init_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_random_init_neighbors_f32";
    std::string_view sv(k_opencl_navatala_vector_search_random_init_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_extract_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_extract_search_results_f32";
    std::string_view sv(k_opencl_navatala_vector_search_extract_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_batched_greedy_search_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_batched_greedy_search_f32";
    std::string_view sv(k_opencl_navatala_vector_search_batched_greedy_search_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_centroid_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_centroid_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_compute_centroid_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scan_inverted_list_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_scan_inverted_list_f32";
    std::string_view sv(k_opencl_navatala_vector_search_scan_inverted_list_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_assign_to_cluster_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_assign_to_cluster_f32";
    std::string_view sv(k_opencl_navatala_vector_search_assign_to_cluster_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_initialize_random_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_initialize_random_graph";
    std::string_view sv(k_opencl_navatala_vector_search_initialize_random_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_graph_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_graph_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_compute_graph_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_graph_r_n_g_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_prune_graph_r_n_g_f32";
    std::string_view sv(k_opencl_navatala_vector_search_prune_graph_r_n_g_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compact_pruned_graph") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compact_pruned_graph";
    std::string_view sv(k_opencl_navatala_vector_search_compact_pruned_graph);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_vertex_degrees") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_vertex_degrees";
    std::string_view sv(k_opencl_navatala_vector_search_compute_vertex_degrees);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_excess_degree") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_prune_excess_degree";
    std::string_view sv(k_opencl_navatala_vector_search_prune_excess_degree);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compact_graph_edges") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compact_graph_edges";
    std::string_view sv(k_opencl_navatala_vector_search_compact_graph_edges);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_validate_graph_integrity") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_validate_graph_integrity";
    std::string_view sv(k_opencl_navatala_vector_search_validate_graph_integrity);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_prune_excess_degree_sorted") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_prune_excess_degree_sorted";
    std::string_view sv(k_opencl_navatala_vector_search_prune_excess_degree_sorted);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_normalize_p_q_codebook_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_normalize_p_q_codebook_f32";
    std::string_view sv(k_opencl_navatala_vector_search_normalize_p_q_codebook_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l2_squared_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_l2_squared_f32";
    std::string_view sv(k_opencl_navatala_vector_search_l2_squared_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_inner_product_f32";
    std::string_view sv(k_opencl_navatala_vector_search_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l1_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_l1_distance_f32";
    std::string_view sv(k_opencl_navatala_vector_search_l1_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_batch_l2_to_neighbors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_batch_l2_to_neighbors_f32";
    std::string_view sv(k_opencl_navatala_vector_search_batch_l2_to_neighbors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_l2_squared_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_l2_squared_f16";
    std::string_view sv(k_opencl_navatala_vector_search_l2_squared_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_cosine_similarity_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_cosine_similarity_f32";
    std::string_view sv(k_opencl_navatala_vector_search_cosine_similarity_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_linf_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_linf_distance_f32";
    std::string_view sv(k_opencl_navatala_vector_search_linf_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_hamming_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_hamming_distance_u32";
    std::string_view sv(k_opencl_navatala_vector_search_hamming_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_correlation_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_correlation_distance_f32";
    std::string_view sv(k_opencl_navatala_vector_search_correlation_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_jaccard_distance_u32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_jaccard_distance_u32";
    std::string_view sv(k_opencl_navatala_vector_search_jaccard_distance_u32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_assign_labels_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_assign_labels_f32";
    std::string_view sv(k_opencl_navatala_vector_search_assign_labels_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_update_centroids_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_update_centroids_f32";
    std::string_view sv(k_opencl_navatala_vector_search_update_centroids_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_cluster_sums_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_clear_cluster_sums_f32";
    std::string_view sv(k_opencl_navatala_vector_search_clear_cluster_sums_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_clear_cluster_counts") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_clear_cluster_counts";
    std::string_view sv(k_opencl_navatala_vector_search_clear_cluster_counts);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_reduce_inertia_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_reduce_inertia_f32";
    std::string_view sv(k_opencl_navatala_vector_search_reduce_inertia_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_bitonic_sort_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_bitonic_sort_step_f32";
    std::string_view sv(k_opencl_navatala_vector_search_bitonic_sort_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_per_row_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_per_row_top_k_f32";
    std::string_view sv(k_opencl_navatala_vector_search_per_row_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_sorted_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_merge_sorted_f32";
    std::string_view sv(k_opencl_navatala_vector_search_merge_sorted_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_rerank_with_exact_distances_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_rerank_with_exact_distances_f32";
    std::string_view sv(k_opencl_navatala_vector_search_rerank_with_exact_distances_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_filter_by_threshold_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_filter_by_threshold_f32";
    std::string_view sv(k_opencl_navatala_vector_search_filter_by_threshold_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_merge_search_results_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_merge_search_results_f32";
    std::string_view sv(k_opencl_navatala_vector_search_merge_search_results_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_recall_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_recall_f32";
    std::string_view sv(k_opencl_navatala_vector_search_compute_recall_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scalar_quantize_f32_to_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_scalar_quantize_f32_to_i8";
    std::string_view sv(k_opencl_navatala_vector_search_scalar_quantize_f32_to_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_scalar_dequantize_i8_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_scalar_dequantize_i8_to_f32";
    std::string_view sv(k_opencl_navatala_vector_search_scalar_dequantize_i8_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_compute_quantized_distances_i8") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_compute_quantized_distances_i8";
    std::string_view sv(k_opencl_navatala_vector_search_compute_quantized_distances_i8);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_encode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_encode_p_q_vectors_f32";
    std::string_view sv(k_opencl_navatala_vector_search_encode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_vector_search_decode_p_q_vectors_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_vector_search_decode_p_q_vectors_f32";
    std::string_view sv(k_opencl_navatala_vector_search_decode_p_q_vectors_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

