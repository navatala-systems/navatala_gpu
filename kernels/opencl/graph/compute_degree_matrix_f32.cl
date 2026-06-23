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

__kernel void navatala_graph_compute_degree_matrix_f32(__global const float* W, __global const uint* n, __global float* D) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
  uint nVal = n[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      float wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint degF32RedStride = (uint)(128u);
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      uint degF32Stride = degF32RedStride;
      if (lid < degF32Stride) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF32StrideToHalve = degF32RedStride;
      uint degF32NextStride = (degF32StrideToHalve >> (uint)(1u));
      degF32RedStride = degF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == (uint)(0u)) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}
