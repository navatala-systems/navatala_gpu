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

__kernel void navatala_vector_search_ivfpq_compute_p_q_distances_f32(__global const uchar* pq_codes, __global const float* lut, __global const uint* vector_indices, __global const uint* list_start, __global const uint* list_size, __global const uint* n_subspaces, __global const uint* n_codes, __global float* distances, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint start = list_start[0];
  uint size = list_size[0];
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  bool inBounds = (gid < size);
  if (inBounds) {
    uint vec_idx = (start + gid);
    float dist = as_float(0x00000000u);
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
