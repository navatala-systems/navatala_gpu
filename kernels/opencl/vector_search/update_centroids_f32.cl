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

__kernel void navatala_vector_search_update_centroids_f32(__global const float* cluster_sums, __global const uint* cluster_counts, __global const uint* k, __global const uint* dim, __global float* centroids) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
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
    bool is_empty = (count == (uint)(0u));
    float safe_count = ((is_empty) ? (as_float(0x3f800000u)) : (count_f));
    float mean = (sum_val / safe_count);
    centroids[gid] = mean;
  }
}
