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
extern "C" __global__ void navatala_vector_search_compute_quantized_distances_i8(const signed char* queries, const signed char* dataset, const unsigned int* n_queries, const unsigned int* n_dataset, const unsigned int* dim, const float* scale, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid_x = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gid_y = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nq = n_queries[0];
  unsigned int nd = n_dataset[0];
  unsigned int d = dim[0];
  float s = scale[0];
  bool in_x = (gid_x < nq);
  bool in_y = (gid_y < nd);
  bool inBounds = (in_x && in_y);
  if (inBounds) {
    int acc = 0;
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_idx = ((gid_x * d) + k);
      unsigned int d_idx = ((gid_y * d) + k);
      signed char q_val = queries[q_idx];
      signed char d_val = dataset[d_idx];
      int q_i32 = ((int)(q_val));
      int d_i32 = ((int)(d_val));
      int diff = (q_i32 - d_i32);
      int sq = (diff * diff);
      int old_acc = acc;
      acc = (old_acc + sq);
    }
    int final_acc = acc;
    float acc_f32 = ((float)(final_acc));
    float result = (acc_f32 * s);
    unsigned int out_idx = ((gid_x * nd) + gid_y);
    distances[out_idx] = result;
  }
}
