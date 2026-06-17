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
extern "C" __global__ void navatala_vector_search_update_centroids_f32(const float* cluster_sums, const unsigned int* cluster_counts, const unsigned int* k, const unsigned int* dim, float* centroids) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  unsigned int d = dim[0];
  unsigned int total = (kVal * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int cluster_idx = (gid / d);
    unsigned int dim_idx = (gid % d);
    unsigned int count = cluster_counts[cluster_idx];
    float sum_val = cluster_sums[gid];
    float count_f = ((float)(count));
    bool is_empty = (count == 0u);
    float safe_count = ((is_empty) ? (__uint_as_float(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}
