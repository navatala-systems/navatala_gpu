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
