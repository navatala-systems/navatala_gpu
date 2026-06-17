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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_mcd_compute_covariance_f64(const double* dataX, const double* dataY, const unsigned int* mask, const double* robustMeanX, const double* robustMeanY, const unsigned int* n, double* robustCov, double* determinant) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double scov[256];
  __shared__ double svarX[256];
  __shared__ double svarY[256];
  __shared__ unsigned int scount[256];
  unsigned int len = n[0];
  double mx = robustMeanX[0];
  double my = robustMeanY[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    unsigned int m = mask[gid];
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
    scov[lid] = __longlong_as_double(0x0000000000000000ull);
    svarX[lid] = __longlong_as_double(0x0000000000000000ull);
    svarY[lid] = __longlong_as_double(0x0000000000000000ull);
    scount[lid] = 0u;
  }
  __syncthreads();
  unsigned int cov2ReductionStride = 128u;
  for (int cov2ReductionStep = 0; cov2ReductionStep < (int)(8); ++cov2ReductionStep) {
    unsigned int cov2Stride = cov2ReductionStride;
    if ((lid < cov2Stride)) {
      double otherCov = scov[(lid + cov2Stride)];
      double otherVX = svarX[(lid + cov2Stride)];
      double otherVY = svarY[(lid + cov2Stride)];
      unsigned int otherC = scount[(lid + cov2Stride)];
      double mineCov = scov[lid];
      double mineVX = svarX[lid];
      double mineVY = svarY[lid];
      unsigned int mineC = scount[lid];
      double sumCov = (mineCov + otherCov);
      double sumVX = (mineVX + otherVX);
      double sumVY = (mineVY + otherVY);
      unsigned int sumC = (mineC + otherC);
      scov[lid] = sumCov;
      svarX[lid] = sumVX;
      svarY[lid] = sumVY;
      scount[lid] = sumC;
    }
    unsigned int cov2StrideToHalve = cov2ReductionStride;
    unsigned int cov2NextStride = (cov2StrideToHalve >> 1u);
    cov2ReductionStride = cov2NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double totalCov = scov[0];
    double totalVX = svarX[0];
    double totalVY = svarY[0];
    unsigned int totalCount = scount[0];
    double countF = ((double)(totalCount));
    double safeCount = (((countF > __longlong_as_double(0x0000000000000000ull))) ? (countF) : (__longlong_as_double(0x3ff0000000000000ull)));
    double covVal = (totalCov / safeCount);
    double varX = (totalVX / safeCount);
    double varY = (totalVY / safeCount);
    double covSq = (covVal * covVal);
    double det = ((varX * varY) - covSq);
    robustCov[0] = covVal;
    determinant[0] = det;
  }
}
