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
extern "C" __global__ void navatala_vector_search_compute_centroid_distances_f32(const float* queries, const float* centroids, const unsigned int* n_queries, const unsigned int* n_lists, const unsigned int* dim, float* centroid_distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nq = n_queries[0];
  unsigned int nl = n_lists[0];
  unsigned int d = dim[0];
  unsigned int total_pairs = (nq * nl);
  bool inBounds = (gid < total_pairs);
  if (inBounds) {
    unsigned int query_idx = (gid / nl);
    unsigned int list_idx = (gid % nl);
    unsigned int query_base = (query_idx * d);
    unsigned int centroid_base = (list_idx * d);
    float acc = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int q_offset = (query_base + k);
      unsigned int c_offset = (centroid_base + k);
      float q_val = queries[q_offset];
      float c_val = centroids[c_offset];
      float diff = (q_val - c_val);
      float sq = (diff * diff);
      float old_acc = acc;
      acc = (old_acc + sq);
    }
    float final_acc = acc;
    centroid_distances[gid] = final_acc;
  }
}
