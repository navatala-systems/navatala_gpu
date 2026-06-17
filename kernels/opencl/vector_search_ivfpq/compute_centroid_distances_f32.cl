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

__kernel void navatala_vector_search_ivfpq_compute_centroid_distances_f32(__global const float* queries, __global const float* centroids, __global const uint* n_queries, __global const uint* n_lists, __global const uint* dim, __global float* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid_q = ((uint)((int)(get_global_id(0))));
  uint gid_c = ((uint)((int)(get_global_id(1))));
  uint nq = n_queries[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool in_q = (gid_q < nq);
  bool in_c = (gid_c < nl);
  bool inBounds = (in_q && in_c);
  if (inBounds) {
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint q_idx = ((gid_q * d) + k);
      uint c_idx = ((gid_c * d) + k);
      float q_val = queries[q_idx];
      float c_val = centroids[c_idx];
      float diff = (q_val - c_val);
      float diff_sq = (diff * diff);
      float old = acc;
      acc = (old + diff_sq);
    }
    uint out_idx = ((gid_q * nl) + gid_c);
    float result = acc;
    distances[out_idx] = result;
  }
}
