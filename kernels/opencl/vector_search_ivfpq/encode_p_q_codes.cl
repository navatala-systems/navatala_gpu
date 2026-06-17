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

__kernel void navatala_vector_search_ivfpq_encode_p_q_codes(__global const float* vectors, __global const float* codebook, __global const uint* n_vectors, __global const uint* n_subspaces, __global const uint* n_codes, __global const uint* sub_dim, __global uchar* codes) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint ns = n_subspaces[0];
  uint nc = n_codes[0];
  uint sd = sub_dim[0];
  uint total = (nv * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint vec_idx = (gid / ns);
    uint subspace = (gid % ns);
    uint dim = (ns * sd);
    uint v_start = ((vec_idx * dim) + (subspace * sd));
    uint cb_subspace_base = (subspace * (nc * sd));
    uint best_code = (uint)(0u);
    float best_dist = as_float(0x7e967699u);
    for (int c = 0; c < (int)(nc); ++c) {
      uint cb_base = (cb_subspace_base + (c * sd));
      float dist = as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        uint v_idx = (v_start + k);
        uint cb_idx = (cb_base + k);
        float v_val = vectors[v_idx];
        float cb_val = codebook[cb_idx];
        float diff = (v_val - cb_val);
        float diff_sq = (diff * diff);
        float old_d = dist;
        dist = (old_d + diff_sq);
      }
      float curr_dist = dist;
      float curr_best = best_dist;
      bool is_better = (curr_dist < curr_best);
      if (is_better) {
        best_dist = curr_dist;
        best_code = c;
      }
    }
    uint final_code = best_code;
    uchar code_u8 = ((uchar)(final_code));
    codes[gid] = code_u8;
  }
}
