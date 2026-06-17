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

__kernel void navatala_vector_search_hamming_distance_u32(__global const uint* x, __global const uint* y, __global const uint* n_x, __global const uint* n_y, __global const uint* dim_words, __global uint* dist) {
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
    uint count = (uint)(0u);
    for (int k = 0; k < (int)(dw); ++k) {
      uint x_idx = ((gid_x * dw) + k);
      uint y_idx = ((gid_y * dw) + k);
      uint x_val = x[x_idx];
      uint y_val = y[y_idx];
      uint xor_val = (x_val ^ y_val);
      uint v0 = xor_val;
      uint m1 = (uint)(1431655765u);
      uint m2 = (uint)(858993459u);
      uint m4 = (uint)(252645135u);
      uint v1 = (v0 - ((v0 >> (uint)(1u)) & m1));
      uint v2 = ((v1 & m2) + ((v1 >> (uint)(2u)) & m2));
      uint v3 = ((v2 + (v2 >> (uint)(4u))) & m4);
      uint v4 = (v3 + (v3 >> (uint)(8u)));
      uint v5 = (v4 + (v4 >> (uint)(16u)));
      uint popcount = (v5 & (uint)(63u));
      uint old_count = count;
      count = (old_count + popcount);
    }
    uint result = count;
    uint out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}
