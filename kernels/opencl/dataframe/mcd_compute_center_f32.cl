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

__kernel void navatala_dataframe_mcd_compute_center_f32(__global const float* dataX, __global const float* dataY, __global const uint* mask, __global const uint* n, __global float* robustMeanX, __global float* robustMeanY) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdataX[256];
  __local float sdataY[256];
  __local uint scount[256];
  uint len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    uint m = mask[gid];
    float x = dataX[gid];
    float y = dataY[gid];
    float mF = ((float)(m));
    float maskedX = (x * mF);
    float maskedY = (y * mF);
    sdataX[lid] = maskedX;
    sdataY[lid] = maskedY;
    scount[lid] = m;
  } else {
    sdataX[lid] = as_float(0x00000000u);
    sdataY[lid] = as_float(0x00000000u);
    scount[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ctr1ReductionStride = (uint)(128u);
  for (int ctr1ReductionStep = 0; ctr1ReductionStep < (int)(8); ++ctr1ReductionStep) {
    uint ctr1Stride = ctr1ReductionStride;
    if ((lid < ctr1Stride)) {
      float otherX = sdataX[(lid + ctr1Stride)];
      float otherY = sdataY[(lid + ctr1Stride)];
      uint otherC = scount[(lid + ctr1Stride)];
      float mineX = sdataX[lid];
      float mineY = sdataY[lid];
      uint mineC = scount[lid];
      float sumX = (mineX + otherX);
      float sumY = (mineY + otherY);
      uint sumC = (mineC + otherC);
      sdataX[lid] = sumX;
      sdataY[lid] = sumY;
      scount[lid] = sumC;
    }
    uint ctr1StrideToHalve = ctr1ReductionStride;
    uint ctr1NextStride = (ctr1StrideToHalve >> (uint)(1u));
    ctr1ReductionStride = ctr1NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    float totalX = sdataX[0];
    float totalY = sdataY[0];
    uint totalCount = scount[0];
    float countF = ((float)(totalCount));
    float safeCount = (((countF > as_float(0x00000000u))) ? (countF) : (as_float(0x3f800000u)));
    float meanX = (totalX / safeCount);
    float meanY = (totalY / safeCount);
    robustMeanX[0] = meanX;
    robustMeanY[0] = meanY;
  }
}
