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

__kernel void navatala_vector_search_assign_to_cluster_f32(__global const float* vectors, __global const float* centroids, __global const uint* n_vectors, __global const uint* n_lists, __global const uint* dim, __global uint* labels) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint vec_base = (gid * d);
    float best_dist = as_float(0x7f7fc99eu);
    uint best_label = (uint)(0u);
    for (int c = 0; c < (int)(nl); ++c) {
      uint cent_base = (c * d);
      float dist_acc = as_float(0x00000000u);
      for (int k = 0; k < (int)(d); ++k) {
        uint v_off = (vec_base + k);
        uint c_off = (cent_base + k);
        float v_val = vectors[v_off];
        float c_val = centroids[c_off];
        float diff = (v_val - c_val);
        float sq = (diff * diff);
        float old_da = dist_acc;
        dist_acc = (old_da + sq);
      }
      float final_da = dist_acc;
      float cur_best = best_dist;
      bool better = (final_da < cur_best);
      if (better) {
        best_dist = final_da;
        best_label = c;
      }
    }
    uint final_label = best_label;
    labels[gid] = final_label;
  }
}
