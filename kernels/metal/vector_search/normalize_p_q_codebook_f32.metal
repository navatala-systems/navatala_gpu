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

kernel void navatala_vector_search_normalize_p_q_codebook_f32(device const int* codebook_sums_i32 [[buffer(0)]], device const uint* codebook_counts [[buffer(1)]], device const uint* n_codewords [[buffer(2)]], device const uint* sub_dim [[buffer(3)]], device float* codebook [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  uint total = (nc * sd);
  float scale = as_type<float>(0x47800000u);
  if ((tid < total)) {
    uint cw = (tid / sd);
    uint count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > as_type<float>(0x00000000u))) ? ((sum_f32 / count_f32)) : (as_type<float>(0x00000000u)));
    codebook[tid] = _centroid;
  }
}
