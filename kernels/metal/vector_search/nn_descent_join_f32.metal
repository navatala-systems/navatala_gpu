// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
        if (((candidate != node_idx) && true)) {
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
