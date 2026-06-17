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
extern "C" __global__ void navatala_vector_search_ivfpq_compute_p_q_distances_f32(const unsigned char* pq_codes, const float* lut, const unsigned int* vector_indices, const unsigned int* list_start, const unsigned int* list_size, const unsigned int* n_subspaces, const unsigned int* n_codes, float* distances, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int start = list_start[0];
  unsigned int size = list_size[0];
  unsigned int ns = n_subspaces[0];
  unsigned int nc = n_codes[0];
  bool inBounds = (gid < size);
  if (inBounds) {
    unsigned int vec_idx = (start + gid);
    float dist = __uint_as_float(0x00000000u);
    for (int s = 0; s < (int)(ns); ++s) {
      unsigned int code_idx = ((vec_idx * ns) + s);
      unsigned char code_u8 = pq_codes[code_idx];
      unsigned int code = ((unsigned int)(code_u8));
      unsigned int lut_idx = ((s * nc) + code);
      float lut_val = lut[lut_idx];
      float old = dist;
      dist = (old + lut_val);
    }
    unsigned int orig_idx = vector_indices[vec_idx];
    float result = dist;
    distances[gid] = result;
    indices[gid] = orig_idx;
  }
}
