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
extern "C" __global__ void navatala_vector_search_reduce_inertia_f32(const float* min_distances, const unsigned int* n_points, float* partial_sums) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int group_id = ((unsigned int)((int)(blockIdx.x)));
  unsigned int n = n_points[0];
  __shared__ float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (__uint_as_float(0x00000000u)));
  sdata[lid] = val;
  __syncthreads();
  unsigned int stride = 128u;
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!(stride > 0u)) break;
    unsigned int s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      unsigned int other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    __syncthreads();
    stride = (s / 2u);
  }
  bool is_thread0 = (lid == 0u);
  if (is_thread0) {
    float block_sum = sdata[0u];
    partial_sums[group_id] = block_sum;
  }
}
