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
__kernel void navatala_dataframe_mcd_compute_center_f64(__global const double* dataX, __global const double* dataY, __global const uint* mask, __global const uint* n, __global double* robustMeanX, __global double* robustMeanY) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdataX[256];
  __local double sdataY[256];
  __local uint scount[256];
  uint len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    uint m = mask[gid];
    double x = dataX[gid];
    double y = dataY[gid];
    double mF = ((double)(m));
    double maskedX = (x * mF);
    double maskedY = (y * mF);
    sdataX[lid] = maskedX;
    sdataY[lid] = maskedY;
    scount[lid] = m;
  } else {
    sdataX[lid] = as_double(0x0000000000000000ul);
    sdataY[lid] = as_double(0x0000000000000000ul);
    scount[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint ctr2ReductionStride = (uint)(128u);
  for (int ctr2ReductionStep = 0; ctr2ReductionStep < (int)(8); ++ctr2ReductionStep) {
    uint ctr2Stride = ctr2ReductionStride;
    if (lid < ctr2Stride) {
      double otherX = sdataX[(lid + ctr2Stride)];
      double otherY = sdataY[(lid + ctr2Stride)];
      uint otherC = scount[(lid + ctr2Stride)];
      double mineX = sdataX[lid];
      double mineY = sdataY[lid];
      uint mineC = scount[lid];
      double sumX = (mineX + otherX);
      double sumY = (mineY + otherY);
      uint sumC = (mineC + otherC);
      sdataX[lid] = sumX;
      sdataY[lid] = sumY;
      scount[lid] = sumC;
    }
    uint ctr2StrideToHalve = ctr2ReductionStride;
    uint ctr2NextStride = (ctr2StrideToHalve >> (uint)(1u));
    ctr2ReductionStride = ctr2NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double totalX = sdataX[0];
    double totalY = sdataY[0];
    uint totalCount = scount[0];
    double countF = ((double)(totalCount));
    double safeCount = (((countF > as_double(0x0000000000000000ul))) ? (countF) : (as_double(0x3ff0000000000000ul)));
    double meanX = (totalX / safeCount);
    double meanY = (totalY / safeCount);
    robustMeanX[0] = meanX;
    robustMeanY[0] = meanY;
  }
}
