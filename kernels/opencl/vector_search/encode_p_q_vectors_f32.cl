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

__kernel void navatala_vector_search_encode_p_q_vectors_f32(__global const float* subvectors, __global const float* codebook, __global const uint* n_vectors, __global const uint* n_codewords, __global const uint* sub_dim, __global uchar* codes) {
  int gid0 = (int)get_global_id(0);
  uint vid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint nc = n_codewords[0];
  uint sd = sub_dim[0];
  if ((vid < nv)) {
    uint best_cw = (uint)(0u);
    float best_dist = as_float(0x7e967699u);
    for (int cw = 0; cw < (int)(nc); ++cw) {
      float dist = as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        uint sv_idx = ((vid * sd) + k);
        uint cb_idx = ((cw * sd) + k);
        float sv = subvectors[sv_idx];
        float cb = codebook[cb_idx];
        float diff = (sv - cb);
        float sq = (diff * diff);
        float old_dist = dist;
        dist = (old_dist + sq);
      }
      float cw_dist = dist;
      float curr_best = best_dist;
      if ((cw_dist < curr_best)) {
        best_cw = cw;
        best_dist = cw_dist;
      }
    }
    uint final_cw = best_cw;
    uchar code = ((uchar)(final_cw));
    codes[vid] = code;
  }
}
