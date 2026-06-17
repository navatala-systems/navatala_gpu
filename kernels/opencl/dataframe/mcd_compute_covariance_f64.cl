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
__kernel void navatala_dataframe_mcd_compute_covariance_f64(__global const double* dataX, __global const double* dataY, __global const uint* mask, __global const double* robustMeanX, __global const double* robustMeanY, __global const uint* n, __global double* robustCov, __global double* determinant) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double scov[256];
  __local double svarX[256];
  __local double svarY[256];
  __local uint scount[256];
  uint len = n[0];
  double mx = robustMeanX[0];
  double my = robustMeanY[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    uint m = mask[gid];
    double x = dataX[gid];
    double y = dataY[gid];
    double mF = ((double)(m));
    double dx = (x - mx);
    double dy = (y - my);
    double covProd = (mF * (dx * dy));
    double varXProd = (mF * (dx * dx));
    double varYProd = (mF * (dy * dy));
    scov[lid] = covProd;
    svarX[lid] = varXProd;
    svarY[lid] = varYProd;
    scount[lid] = m;
  } else {
    scov[lid] = as_double(0x0000000000000000ul);
    svarX[lid] = as_double(0x0000000000000000ul);
    svarY[lid] = as_double(0x0000000000000000ul);
    scount[lid] = (uint)(0u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cov2ReductionStride = (uint)(128u);
  for (int cov2ReductionStep = 0; cov2ReductionStep < (int)(8); ++cov2ReductionStep) {
    uint cov2Stride = cov2ReductionStride;
    if ((lid < cov2Stride)) {
      double otherCov = scov[(lid + cov2Stride)];
      double otherVX = svarX[(lid + cov2Stride)];
      double otherVY = svarY[(lid + cov2Stride)];
      uint otherC = scount[(lid + cov2Stride)];
      double mineCov = scov[lid];
      double mineVX = svarX[lid];
      double mineVY = svarY[lid];
      uint mineC = scount[lid];
      double sumCov = (mineCov + otherCov);
      double sumVX = (mineVX + otherVX);
      double sumVY = (mineVY + otherVY);
      uint sumC = (mineC + otherC);
      scov[lid] = sumCov;
      svarX[lid] = sumVX;
      svarY[lid] = sumVY;
      scount[lid] = sumC;
    }
    uint cov2StrideToHalve = cov2ReductionStride;
    uint cov2NextStride = (cov2StrideToHalve >> (uint)(1u));
    cov2ReductionStride = cov2NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double totalCov = scov[0];
    double totalVX = svarX[0];
    double totalVY = svarY[0];
    uint totalCount = scount[0];
    double countF = ((double)(totalCount));
    double safeCount = (((countF > as_double(0x0000000000000000ul))) ? (countF) : (as_double(0x3ff0000000000000ul)));
    double covVal = (totalCov / safeCount);
    double varX = (totalVX / safeCount);
    double varY = (totalVY / safeCount);
    double covSq = (covVal * covVal);
    double det = ((varX * varY) - covSq);
    robustCov[0] = covVal;
    determinant[0] = det;
  }
}
