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
