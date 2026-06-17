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
