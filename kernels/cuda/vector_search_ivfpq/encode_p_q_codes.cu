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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_vector_search_ivfpq_encode_p_q_codes(const float* vectors, const float* codebook, const unsigned int* n_vectors, const unsigned int* n_subspaces, const unsigned int* n_codes, const unsigned int* sub_dim, unsigned char* codes) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nv * ns);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int vec_idx = (gid / ns);
    unsigned int subspace = (gid % ns);
    unsigned int dim = (ns * sd);
    unsigned int v_start = ((vec_idx * dim) + (subspace * sd));
    unsigned int cb_subspace_base = (subspace * (nc * sd));
    unsigned int best_code = 0u;
    float best_dist = __uint_as_float(0x7e967699u);
    for (int c = 0; c < (int)(nc); ++c) {
      unsigned int cb_base = (cb_subspace_base + (c * sd));
      float dist = __uint_as_float(0x00000000u);
      for (int k = 0; k < (int)(sd); ++k) {
        unsigned int v_idx = (v_start + k);
        unsigned int cb_idx = (cb_base + k);
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
    unsigned int final_code = best_code;
    unsigned char code_u8 = ((unsigned char)(final_code));
    codes[gid] = code_u8;
  }
}
