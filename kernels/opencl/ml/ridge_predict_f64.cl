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
__kernel void navatala_ml_ridge_predict_f64(__global const double* X, __global const double* beta, __global const uint* n, __global const uint* p, __global double* yPred) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint pVal = p[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      uint elemIdx = ((rowIdx * pVal) + colIdx);
      double xVal = X[elemIdx];
      double betaVal = beta[colIdx];
      double product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint pred64_reductionStride = (uint)(128u);
    for (int pred64_reductionStep = 0; pred64_reductionStep < (int)(8); ++pred64_reductionStep) {
      uint pred64_stride = pred64_reductionStride;
      if ((lid < pred64_stride)) {
        double pred64_other = sdata[(lid + pred64_stride)];
        double pred64_mine = sdata[lid];
        double pred64_sum = (pred64_mine + pred64_other);
        sdata[lid] = pred64_sum;
      }
      uint pred64_strideToHalve = pred64_reductionStride;
      uint pred64_nextStride = (pred64_strideToHalve >> (uint)(1u));
      pred64_reductionStride = pred64_nextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}
