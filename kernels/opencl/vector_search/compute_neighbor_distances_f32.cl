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

__kernel void navatala_vector_search_compute_neighbor_distances_f32(__global const float* query, __global const float* dataset, __global const uint* neighbor_ids, __global const uint* n_neighbors, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint tid = ((uint)((int)(get_global_id(0))));
  uint nn = n_neighbors[0];
  uint d = dim[0];
  if (tid < nn) {
    uint neighbor_id = neighbor_ids[tid];
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      float q_val = query[k];
      uint n_idx = ((neighbor_id * d) + k);
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
