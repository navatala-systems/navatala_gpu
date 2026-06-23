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
extern "C" __global__ void navatala_vector_search_accumulate_p_q_codebook_i32(const float* subvectors, const unsigned int* assignments, const unsigned int* n_vectors, const unsigned int* sub_dim, int* codebook_sums_i32, unsigned int* codebook_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int vid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int sd = sub_dim[0];
  float scale = __uint_as_float(0x47800000u);
  if (vid < nv) {
    unsigned int cw = assignments[vid];
    for (int k = 0; k < (int)(sd); ++k) {
      unsigned int src_idx = ((vid * sd) + k);
      unsigned int dst_idx = ((cw * sd) + k);
      float val_f32 = subvectors[src_idx];
      float scaled_f32 = (val_f32 * scale);
      int scaled_i32 = ((int)(scaled_f32));
      int _old = atomicAdd(&(codebook_sums_i32[dst_idx]), scaled_i32);
    }
    unsigned int _cnt = atomicAdd(&(codebook_counts[cw]), 1u);
  }
}
