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

__kernel void navatala_vector_search_reduce_inertia_f32(__global const float* min_distances, __global const uint* n_points, __global float* partial_sums) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint group_id = ((uint)((int)(get_group_id(0))));
  uint n = n_points[0];
  __local float sdata[256];
  bool inBounds = (gid < n);
  float val = ((inBounds) ? (min_distances[gid]) : (as_float(0x00000000u)));
  sdata[lid] = val;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride = (uint)(128u);
  for (int __iter = 0; __iter < 8; ++__iter) {
    if (!(stride > (uint)(0u))) break;
    uint s = stride;
    bool should_reduce = (lid < s);
    if (should_reduce) {
      uint other_idx = (lid + s);
      float my_val = sdata[lid];
      float other_val = sdata[other_idx];
      float sum = (my_val + other_val);
      sdata[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    stride = (s / (uint)(2u));
  }
  bool is_thread0 = (lid == (uint)(0u));
  if (is_thread0) {
    float block_sum = sdata[(uint)(0u)];
    partial_sums[group_id] = block_sum;
  }
}
