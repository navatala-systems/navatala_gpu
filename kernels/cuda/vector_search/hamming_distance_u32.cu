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
extern "C" __global__ void navatala_vector_search_hamming_distance_u32(const unsigned int* x, const unsigned int* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim_words, unsigned int* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int dw = dim_words[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    unsigned int count = 0u;
    for (int k = 0; k < (int)(dw); ++k) {
      unsigned int x_idx = ((gid_x * dw) + k);
      unsigned int y_idx = ((gid_y * dw) + k);
      unsigned int x_val = x[x_idx];
      unsigned int y_val = y[y_idx];
      unsigned int xor_val = (x_val ^ y_val);
      unsigned int v0 = xor_val;
      unsigned int m1 = 1431655765u;
      unsigned int m2 = 858993459u;
      unsigned int m4 = 252645135u;
      unsigned int v1 = (v0 - ((v0 >> 1u) & m1));
      unsigned int v2 = ((v1 & m2) + ((v1 >> 2u) & m2));
      unsigned int v3 = ((v2 + (v2 >> 4u)) & m4);
      unsigned int v4 = (v3 + (v3 >> 8u));
      unsigned int v5 = (v4 + (v4 >> 16u));
      unsigned int popcount = (v5 & 63u);
      unsigned int old_count = count;
      count = (old_count + popcount);
    }
    unsigned int result = count;
    unsigned int out_idx = ((gid_x * ny) + gid_y);
    dist[out_idx] = result;
  }
}
