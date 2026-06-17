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

kernel void navatala_vector_search_ivfpq_compute_p_q_distances_f32(device const uchar* pq_codes [[buffer(0)]], device const float* lut [[buffer(1)]], device const uint* vector_indices [[buffer(2)]], device const uint* list_start [[buffer(3)]], device const uint* list_size [[buffer(4)]], device const uint* n_subspaces [[buffer(5)]], device const uint* n_codes [[buffer(6)]], device float* distances [[buffer(7)]], device uint* indices [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint start = list_start[0];
  uint size = list_size[0];
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  bool inBounds = (gid < size);
  if (inBounds) {
    uint vec_idx = (start + gid);
    float dist = as_type<float>(0x00000000u);
    for (int s = 0; s < (int)(ns); ++s) {
      uint code_idx = ((vec_idx * ns) + s);
      uchar code_u8 = pq_codes[code_idx];
      uint code = ((uint)(code_u8));
      uint lut_idx = ((s * nc) + code);
      float lut_val = lut[lut_idx];
      float old = dist;
      dist = (old + lut_val);
    }
    uint orig_idx = vector_indices[vec_idx];
    float result = dist;
    distances[gid] = result;
    indices[gid] = orig_idx;
  }
}
