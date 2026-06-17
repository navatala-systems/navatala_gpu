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

__kernel void navatala_vector_search_jaccard_distance_u32(__global const uint* x, __global const uint* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim_words, __global float* dist) {
  int gid0 = (int)get_global_id(0);
  uint gid_x = ((uint)((int)(get_global_id(0))));
  uint gid_y = ((uint)((int)(get_global_id(1))));
  uint nx = n_x[0];
  uint ny = n_y[0];
  uint dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    uint intersection = (uint)(0u);
    uint union_count = (uint)(0u);
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint inter = (x_val & y_val);
      uint uni = (x_val | y_val);
      uint i0 = inter;
      uint m1 = (uint)(1431655765u);
      uint m2 = (uint)(858993459u);
      uint m4 = (uint)(252645135u);
      uint i1 = (i0 - ((i0 >> (uint)(1u)) & m1));
      uint i2 = ((i1 & m2) + ((i1 >> (uint)(2u)) & m2));
      uint i3 = ((i2 + (i2 >> (uint)(4u))) & m4);
      uint i4 = (i3 + (i3 >> (uint)(8u)));
      uint i5 = (i4 + (i4 >> (uint)(16u)));
      uint pop_inter = (i5 & (uint)(63u));
      uint u0 = uni;
      uint u1 = (u0 - ((u0 >> (uint)(1u)) & m1));
      uint u2 = ((u1 & m2) + ((u1 >> (uint)(2u)) & m2));
      uint u3 = ((u2 + (u2 >> (uint)(4u))) & m4);
      uint u4 = (u3 + (u3 >> (uint)(8u)));
      uint u5 = (u4 + (u4 >> (uint)(16u)));
      uint pop_union = (u5 & (uint)(63u));
      uint old_inter = intersection;
      uint old_union = union_count;
      intersection = (old_inter + pop_inter);
      union_count = (old_union + pop_union);
    }
    uint final_inter = intersection;
    uint final_union = union_count;
    float inter_f = ((float)(final_inter));
    float union_f = ((float)(final_union));
    float eps = as_float(0x322bcc77u);
    float safe_union = (union_f + eps);
    float similarity = (inter_f / safe_union);
    float jaccard_dist = (as_float(0x3f800000u) - similarity);
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = jaccard_dist;
  }
}
