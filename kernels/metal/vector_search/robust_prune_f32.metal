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
