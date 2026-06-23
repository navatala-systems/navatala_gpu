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

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_graph_spmv_weighted_subgroup_f32(__global const uint* offsets, __global const uint* indices, __global const float* weights, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lane = ((uint)((int)(get_sub_group_local_id())));
  uint subgroupSize = ((uint)((int)(get_sub_group_size())));
  uint row = (gid / subgroupSize);
  uint numV = numVertices[0];
  if (row < numV) {
    uint base = offsets[row];
    uint endv = offsets[(row + (uint)(1u))];
    uint rowlen = (endv - base);
    uint rowIters = ((rowlen + (subgroupSize - (uint)(1u))) / subgroupSize);
    float laneAcc = as_float(0x00000000u);
    for (int k = 0; k < (int)(rowIters); ++k) {
      uint rel = ((((uint)(k)) * subgroupSize) + lane);
      if (rel < rowlen) {
        uint eidx = (base + rel);
        uint col = indices[eidx];
        float w = weights[eidx];
        float xv = x[col];
        laneAcc = (laneAcc + (w * xv));
      }
    }
    float rowSum = sub_group_reduce_add(laneAcc);
    if (lane == (uint)(0u)) {
      y[row] = rowSum;
    }
  }
}
