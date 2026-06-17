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
extern "C" __global__ void navatala_vector_search_accumulate_cluster_sums_f32(const float* data, const unsigned int* labels, const unsigned int* n_points, const unsigned int* k, const unsigned int* dim, float* cluster_sums, unsigned int* cluster_counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = n_points[0];
  unsigned int d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int label = labels[gid];
    unsigned int _old_count = atomicAdd(&(cluster_counts[label]), 1u);
    for (int j = 0; j < (int)(d); ++j) {
      unsigned int data_idx = ((gid * d) + j);
      float data_val = data[data_idx];
      unsigned int sum_idx = ((label * d) + j);
      atomicAdd(&cluster_sums[sum_idx], data_val);
    }
  }
}
