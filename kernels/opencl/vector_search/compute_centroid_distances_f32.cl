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

__kernel void navatala_vector_search_compute_centroid_distances_f32(__global const float* queries, __global const float* centroids, __global const uint* n_queries, __global const uint* n_lists, __global const uint* dim, __global float* centroid_distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nq = n_queries[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  uint total_pairs = (nq * nl);
  bool inBounds = (gid < total_pairs);
  if (inBounds) {
    uint query_idx = (gid / nl);
    uint list_idx = (gid % nl);
    uint query_base = (query_idx * d);
    uint centroid_base = (list_idx * d);
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_offset = (query_base + k);
      uint c_offset = (centroid_base + k);
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
