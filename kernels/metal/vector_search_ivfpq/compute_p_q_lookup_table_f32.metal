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

kernel void navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32(device const float* query [[buffer(0)]], device const float* codebook [[buffer(1)]], device const uint* n_subspaces [[buffer(2)]], device const uint* n_codes [[buffer(3)]], device const uint* sub_dim [[buffer(4)]], device float* lut [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  uint sd = sub_dim[0];
  uint total = (ns * nc);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint subspace = (gid / nc);
    uint code = (gid % nc);
    uint q_start = (subspace * sd);
    uint cb_base = ((subspace * (nc * sd)) + (code * sd));
    float dist = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(sd); ++k) {
      uint q_idx = (q_start + k);
      uint cb_idx = (cb_base + k);
      float q_val = query[q_idx];
      float cb_val = codebook[cb_idx];
      float diff = (q_val - cb_val);
      float diff_sq = (diff * diff);
      float old = dist;
      dist = (old + diff_sq);
    }
    float result = dist;
    lut[gid] = result;
  }
}
