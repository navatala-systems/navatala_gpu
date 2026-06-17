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

kernel void navatala_vector_search_jaccard_distance_u32(device const uint* x [[buffer(0)]], device const uint* y [[buffer(1)]], device const uint* n_x [[buffer(2)]], device const uint* n_y [[buffer(3)]], device const uint* dim_words [[buffer(4)]], device float* dist [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid_x = ((uint)(int(__gid.x)));
  uint gid_y = ((uint)(int(__gid.y)));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint intersection = 0u;
    uint union_count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint inter = (x_val & y_val);
      uint uni = (x_val | y_val);
      uint i0 = inter;
      uint m1 = 1431655765u;
      uint m2 = 858993459u;
      uint m4 = 252645135u;
      uint i1 = (i0 - ((i0 >> 1u) & m1));
      uint i2 = ((i1 & m2) + ((i1 >> 2u) & m2));
      uint i3 = ((i2 + (i2 >> 4u)) & m4);
      uint i4 = (i3 + (i3 >> 8u));
      uint i5 = (i4 + (i4 >> 16u));
      uint pop_inter = (i5 & 63u);
      uint u0 = uni;
      uint u1 = (u0 - ((u0 >> 1u) & m1));
      uint u2 = ((u1 & m2) + ((u1 >> 2u) & m2));
      uint u3 = ((u2 + (u2 >> 4u)) & m4);
      uint u4 = (u3 + (u3 >> 8u));
      uint u5 = (u4 + (u4 >> 16u));
      uint pop_union = (u5 & 63u);
      uint old_inter = intersection;
      uint old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    uint final_inter = intersection;
    uint final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = as_type<float>(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (as_type<float>(0x3f800000u) - similarity);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}
