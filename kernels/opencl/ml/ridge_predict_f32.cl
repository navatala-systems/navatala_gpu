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

__kernel void navatala_ml_ridge_predict_f32(__global const float* X, __global const float* beta, __global const uint* n, __global const uint* p, __global float* yPred) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local float sdata[256];
  uint nVal = n[0];
  uint pVal = p[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      uint elemIdx = ((rowIdx * pVal) + colIdx);
      float xVal = X[elemIdx];
      float betaVal = beta[colIdx];
      float product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_float(0x00000000u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint pred32_reductionStride = (uint)(128u);
    for (int pred32_reductionStep = 0; pred32_reductionStep < (int)(8); ++pred32_reductionStep) {
      uint pred32_stride = pred32_reductionStride;
      if ((lid < pred32_stride)) {
        float pred32_other = sdata[(lid + pred32_stride)];
        float pred32_mine = sdata[lid];
        float pred32_sum = (pred32_mine + pred32_other);
        sdata[lid] = pred32_sum;
      }
      uint pred32_strideToHalve = pred32_reductionStride;
      uint pred32_nextStride = (pred32_strideToHalve >> (uint)(1u));
      pred32_reductionStride = pred32_nextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      float dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}
