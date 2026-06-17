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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_encode_p_q_vectors_f32(const float* subvectors, const float* codebook, const unsigned int* n_vectors, const unsigned int* n_codewords, const unsigned int* sub_dim, unsigned char* codes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int nc = n_codewords[0];
  unsigned int sd = sub_dim[0];
  if ((vid < nv)) {
    unsigned int best_cw = 0u;
    float best_dist = __uint_as_float(0x7e967699u);
    for (int cw = 0; cw < (int)(nc); ++cw) {
      float dist = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        unsigned int sv_idx = ((vid * sd) + k);
        unsigned int cb_idx = ((cw * sd) + k);
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
    unsigned int final_cw = best_cw;
    unsigned char code = ((unsigned char)(final_cw));
    codes[vid] = code;
  }
}
