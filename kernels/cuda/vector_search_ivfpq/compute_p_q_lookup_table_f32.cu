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
extern "C" __global__ void navatala_vector_search_ivfpq_compute_p_q_lookup_table_f32(const float* query, const float* codebook, const unsigned int* n_subspaces, const unsigned int* n_codes, const unsigned int* sub_dim, float* lut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (ns * nc);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int subspace = (gid / nc);
    unsigned int code = (gid % nc);
    unsigned int q_start = (subspace * sd);
    unsigned int cb_base = ((subspace * (nc * sd)) + (code * sd));
    float dist = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(sd); ++k) {
      unsigned int q_idx = (q_start + k);
      unsigned int cb_idx = (cb_base + k);
      float q_val = query[q_idx];
      float cb_val = codebook[cb_idx];
      float diff = (q_val - cb_val);
      float diff_sq = (diff * diff);
      float old = dist;
      dist = (old + diff_sq);
    }
    float result = dist;
    lut[gid] = result;
  }
}
