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
__kernel void navatala_graph_normalize_eigenvector_rows_f64(__global const double* V, __global const uint* n, __global const uint* k, __global double* V_norm) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint kVal = k[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      double vVal = V[elemIdx];
      double vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint normF64RedStride = (uint)(128u);
    for (int normF64RedStep = 0; normF64RedStep < (int)(8); ++normF64RedStep) {
      uint normF64Stride = normF64RedStride;
      if (lid < normF64Stride) {
        double other = sdata[(lid + normF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      uint normF64StrideToHalve = normF64RedStride;
      uint normF64NextStride = (normF64StrideToHalve >> (uint)(1u));
      normF64RedStride = normF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == (uint)(0u)) {
      double sumSq = sdata[0];
      double normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (colInBounds) {
      double norm = sdata[0];
      double vVal2 = V[elemIdx];
      double eps = as_double(0x3cd203af9ee75616ul);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = as_double(0x0000000000000000ul);
      } else {
        double normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}
