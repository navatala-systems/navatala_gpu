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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_vector_search_assign_to_cluster_f32(device const float* vectors [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_vectors [[buffer(2)]], device const uint* n_lists [[buffer(3)]], device const uint* dim [[buffer(4)]], device uint* labels [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nv = n_vectors[0];
  uint nl = n_lists[0];
  uint d = dim[0];
  bool inBounds = (gid < nv);
  if (inBounds) {
    uint vec_base = (gid * d);
    float best_dist = as_type<float>(0x7f7fc99eu);
    uint best_label = 0u;
    for (int c = 0; c < (int)(nl); ++c) {
      uint cent_base = (c * d);
      float dist_acc = as_type<float>(0x00000000u);
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
