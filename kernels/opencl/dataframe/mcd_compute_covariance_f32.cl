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

__kernel void navatala_dataframe_mcd_compute_covariance_f32(__global const float* dataX, __global const float* dataY, __global const uint* mask, __global const float* robustMeanX, __global const float* robustMeanY, __global const uint* n, __global float* robustCov, __global float* determinant) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float scov[256];
  __local float svarX[256];
  __local float svarY[256];
  __local uint scount[256];
  uint len = n[0];
  float mx = robustMeanX[0];
  float my = robustMeanY[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    uint m = mask[gid];
    float x = dataX[gid];
    float y = dataY[gid];
    float mF = ((float)(m));
    float dx = (x - mx);
    float dy = (y - my);
    float covProd = (mF * (dx * dy));
    float varXProd = (mF * (dx * dx));
    float varYProd = (mF * (dy * dy));
    scov[lid] = covProd;
    svarX[lid] = varXProd;
    svarY[lid] = varYProd;
    scount[lid] = m;
  } else {
    scov[lid] = as_float(0x00000000u);
    svarX[lid] = as_float(0x00000000u);
    svarY[lid] = as_float(0x00000000u);
    scount[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cov1ReductionStride = (uint)(128u);
  for (int cov1ReductionStep = 0; cov1ReductionStep < (int)(8); ++cov1ReductionStep) {
    uint cov1Stride = cov1ReductionStride;
    if (lid < cov1Stride) {
      float otherCov = scov[(lid + cov1Stride)];
      float otherVX = svarX[(lid + cov1Stride)];
      float otherVY = svarY[(lid + cov1Stride)];
      uint otherC = scount[(lid + cov1Stride)];
      float mineCov = scov[lid];
      float mineVX = svarX[lid];
      float mineVY = svarY[lid];
      uint mineC = scount[lid];
      float sumCov = (mineCov + otherCov);
      float sumVX = (mineVX + otherVX);
      float sumVY = (mineVY + otherVY);
      uint sumC = (mineC + otherC);
      scov[lid] = sumCov;
      svarX[lid] = sumVX;
      svarY[lid] = sumVY;
      scount[lid] = sumC;
    }
    uint cov1StrideToHalve = cov1ReductionStride;
    uint cov1NextStride = (cov1StrideToHalve >> (uint)(1u));
    cov1ReductionStride = cov1NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float totalCov = scov[0];
    float totalVX = svarX[0];
    float totalVY = svarY[0];
    uint totalCount = scount[0];
    float countF = ((float)(totalCount));
    float safeCount = (((countF > as_float(0x00000000u))) ? (countF) : (as_float(0x3f800000u)));
    float covVal = (totalCov / safeCount);
    float varX = (totalVX / safeCount);
    float varY = (totalVY / safeCount);
    float covSq = (covVal * covVal);
    float det = ((varX * varY) - covSq);
    robustCov[0] = covVal;
    determinant[0] = det;
  }
}
