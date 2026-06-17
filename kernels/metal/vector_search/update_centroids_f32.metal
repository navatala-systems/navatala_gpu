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

kernel void navatala_vector_search_update_centroids_f32(device const float* cluster_sums [[buffer(0)]], device const uint* cluster_counts [[buffer(1)]], device const uint* k [[buffer(2)]], device const uint* dim [[buffer(3)]], device float* centroids [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  uint d = dim[0];
  uint total = (kVal * d);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint cluster_idx = (gid / d);
    uint dim_idx = (gid % d);
    uint count = cluster_counts[cluster_idx];
    float sum_val = cluster_sums[gid];
    float count_f = ((float)(count));
    bool is_empty = (count == 0u);
    float safe_count = ((is_empty) ? (as_type<float>(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}
