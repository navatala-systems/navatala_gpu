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
extern "C" __global__ void navatala_vector_search_normalize_p_q_codebook_f32(const int* codebook_sums_i32, const unsigned int* codebook_counts, const unsigned int* n_codewords, const unsigned int* sub_dim, float* codebook) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nc = n_codewords[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nc * sd);
  float scale = __uint_as_float(0x47800000u);
  if (tid < total) {
    unsigned int cw = (tid / sd);
    unsigned int count_u32 = codebook_counts[cw];
    int sum_i32 = codebook_sums_i32[tid];
    float sum_f32 = (((float)(sum_i32)) / scale);
    float count_f32 = ((float)(count_u32));
    float _centroid = (((count_f32 > __uint_as_float(0x00000000u))) ? ((sum_f32 / count_f32)) : (__uint_as_float(0x00000000u)));
    codebook[tid] = _centroid;
  }
}
