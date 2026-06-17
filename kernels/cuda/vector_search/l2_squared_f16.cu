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
#include <cuda_fp16.h>
extern "C" __global__ void navatala_vector_search_l2_squared_f16(const __half* x, const __half* y, const unsigned int* n_x, const unsigned int* n_y, const unsigned int* dim, float* dist) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nx = n_x[0];
  unsigned int ny = n_y[0];
  unsigned int d = dim[0];
  bool in_x = (gid_x < nx);
  bool in_y = (gid_y < ny);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int x_idx = ((gid_x * d) + k);
      unsigned int y_idx = ((gid_y * d) + k);
      __half x_f16 = x[x_idx];
      __half y_f16 = y[y_idx];
      float x_val = ((float)(x_f16));
      float y_val = ((float)(y_f16));
      float diff = (x_val - y_val);
      float diff_sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + diff_sq);
    }
    unsigned int dist_idx = ((gid_x * ny) + gid_y);
    float result = acc;
    dist[dist_idx] = result;
  }
}
