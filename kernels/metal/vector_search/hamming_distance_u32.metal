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

kernel void navatala_vector_search_hamming_distance_u32(device const uint* x [[buffer(0)]], device const uint* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim_words [[buffer(4)]], device uint* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint xor_val = (x_val ^ y_val);
      uint v0 = xor_val;
      uint m1 = 1431655765u;
      uint m2 = 858993459u;
      uint m4 = 252645135u;
      uint v1 = (v0 - ((v0 >> 1u) & m1));
      uint v2 = ((v1 & m2) + ((v1 >> 2u) & m2));
      uint v3 = ((v2 + (v2 >> 4u)) & m4);
      uint v4 = (v3 + (v3 >> 8u));
      uint v5 = (v4 + (v4 >> 16u));
      uint popcount = (v5 & 63u);
      uint old_count = count;
      count = (old_count + popcount);
    }
    uint result = count;
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}
