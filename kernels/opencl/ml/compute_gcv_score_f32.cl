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

__kernel void navatala_ml_compute_gcv_score_f32(__global const float* y, __global const float* yPred, __global const float* traceH, __global const uint* n, __global float* gcv) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint nVal = n[0];
  float traceHVal = traceH[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float residual = (yVal - yPredVal);
    float residualSq = (residual * residual);
    sdata[lid] = residualSq;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint gcv32_reductionStride = (uint)(128u);
  for (int gcv32_reductionStep = 0; gcv32_reductionStep < (int)(8); ++gcv32_reductionStep) {
    uint gcv32_stride = gcv32_reductionStride;
    if (lid < gcv32_stride) {
      float gcv32_other = sdata[(lid + gcv32_stride)];
      float gcv32_mine = sdata[lid];
      float gcv32_sum = (gcv32_mine + gcv32_other);
      sdata[lid] = gcv32_sum;
    }
    uint gcv32_strideToHalve = gcv32_reductionStride;
    uint gcv32_nextStride = (gcv32_strideToHalve >> (uint)(1u));
    gcv32_reductionStride = gcv32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float rss = sdata[0];
    float nFloat = ((float)(nVal));
    float traceHOverN = (traceHVal / nFloat);
    float oneMinusRatio = (as_float(0x3f800000u) - traceHOverN);
    float denom = (oneMinusRatio * oneMinusRatio);
    float fullDenom = (nFloat * denom);
    float gcvVal = (rss / fullDenom);
    gcv[0] = gcvVal;
  }
}
