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

__kernel void navatala_graph_spmv_unweighted_f32(__global const uint* offsets, __global const uint* indices, __global const float* x, __global const uint* numVertices, __global float* y) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + (uint)(1u))];
    uint rowlen = (endv - base);
    float acc = as_float(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint col = indices[eidx];
      float xv = x[col];
      acc = (acc + xv);
    }
    y[gid] = acc;
  }
}
