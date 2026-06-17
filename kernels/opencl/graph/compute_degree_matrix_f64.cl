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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_compute_degree_matrix_f64(__global const double* W, __global const uint* n, __global double* D) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      double wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint degF64RedStride = (uint)(128u);
    for (int degF64RedStep = 0; degF64RedStep < (int)(8); ++degF64RedStep) {
      uint degF64Stride = degF64RedStride;
      if ((lid < degF64Stride)) {
        double other = sdata[(lid + degF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF64StrideToHalve = degF64RedStride;
      uint degF64NextStride = (degF64StrideToHalve >> (uint)(1u));
      degF64RedStride = degF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}
