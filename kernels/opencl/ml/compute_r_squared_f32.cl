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

__kernel void navatala_ml_compute_r_squared_f32(__global const float* y, __global const float* yPred, __global const float* yMean, __global const uint* nSamples, __global float* rSquared) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdataRes[256];
  __local float sdataTot[256];
  uint n = nSamples[0];
  float meanVal = yMean[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float resid = (yVal - yPredVal);
    float residSq = (resid * resid);
    float dev = (yVal - meanVal);
    float devSq = (dev * dev);
    sdataRes[lid] = residSq;
    sdataTot[lid] = devSq;
  } else {
    sdataRes[lid] = as_float(0x00000000u);
    sdataTot[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint r2F32Stride = (uint)(128u);
  for (int reductionStep3 = 0; reductionStep3 < (int)(8); ++reductionStep3) {
    uint stride3 = r2F32Stride;
    if ((lid < stride3)) {
      float otherRes = sdataRes[(lid + stride3)];
      float mineRes = sdataRes[lid];
      float sumRes = (mineRes + otherRes);
      sdataRes[lid] = sumRes;
      float otherTot = sdataTot[(lid + stride3)];
      float mineTot = sdataTot[lid];
      float sumTot = (mineTot + otherTot);
      sdataTot[lid] = sumTot;
    }
    uint strideToHalve3 = r2F32Stride;
    uint nextStride3 = (strideToHalve3 >> (uint)(1u));
    r2F32Stride = nextStride3;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float ssRes = sdataRes[0];
    float ssTot = sdataTot[0];
    float ratio = (ssRes / ssTot);
    float r2 = (as_float(0x3f800000u) - ratio);
    rSquared[0] = r2;
  }
}
