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
extern "C" __global__ void navatala_vector_search_compute_neighbor_distances_f32(const float* query, const float* dataset, const unsigned int* neighbor_ids, const unsigned int* n_neighbors, const unsigned int* dim, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nn = n_neighbors[0];
  unsigned int d = dim[0];
  if (tid < nn) {
    unsigned int neighbor_id = neighbor_ids[tid];
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      unsigned int n_idx = ((neighbor_id * d) + k);
      float n_val = dataset[n_idx];
      float diff = (q_val - n_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_dist = acc;
    distances[tid] = final_dist;
  }
}
