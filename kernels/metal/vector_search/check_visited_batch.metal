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

kernel void navatala_vector_search_check_visited_batch(device const uint* node_ids [[buffer(0)]], device const uint* query_ids [[buffer(1)]], device const uint* n_nodes [[buffer(2)]], device const uint* visited_stride [[buffer(3)]], device const uint* visited [[buffer(4)]], device uint* is_visited [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nn = n_nodes[0];
  uint stride = visited_stride[0];
  if ((tid < nn)) {
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
