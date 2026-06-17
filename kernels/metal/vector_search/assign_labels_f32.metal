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

kernel void navatala_vector_search_assign_labels_f32(device const float* data [[buffer(0)]], device const float* centroids [[buffer(1)]], device const uint* n_points [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* dim [[buffer(4)]], device uint* labels [[buffer(5)]], device float* min_distances [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = n_points[0];
  uint kVal = k[0];
  uint d = dim[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float best_dist = as_type<float>(0x7e967699u);
    uint best_label = 0u;
    for (int c = 0; c < (int)(kVal); ++c) {
      float dist_acc = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(d); ++j) {
        uint data_idx = ((gid * d) + j);
        float data_val = data[data_idx];
        uint cent_idx = ((c * d) + j);
        float cent_val = centroids[cent_idx];
        float diff = (data_val - cent_val);
        float diff_sq = (diff * diff);
        float old_acc = dist_acc;
        dist_acc = (old_acc + diff_sq);
      }
      float curr_dist = dist_acc;
      float curr_best = best_dist;
      bool is_better = (curr_dist < curr_best);
      if (is_better) {
        best_dist = curr_dist;
        best_label = c;
      }
    }
    uint final_label = best_label;
    float final_dist = best_dist;
    labels[gid] = final_label;
    min_distances[gid] = final_dist;
  }
}
