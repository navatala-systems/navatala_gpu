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
extern "C" __global__ void navatala_dataframe_mcd_compute_covariance_f32(const float* dataX, const float* dataY, const unsigned int* mask, const float* robustMeanX, const float* robustMeanY, const unsigned int* n, float* robustCov, float* determinant) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float scov[256];
  __shared__ float svarX[256];
  __shared__ float svarY[256];
  __shared__ unsigned int scount[256];
  unsigned int len = n[0];
  float mx = robustMeanX[0];
  float my = robustMeanY[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    unsigned int m = mask[gid];
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
    scov[lid] = __uint_as_float(0x00000000u);
    svarX[lid] = __uint_as_float(0x00000000u);
    svarY[lid] = __uint_as_float(0x00000000u);
    scount[lid] = 0u;
  }
  __syncthreads();
  unsigned int cov1ReductionStride = 128u;
  for (int cov1ReductionStep = 0; cov1ReductionStep < (int)(8); ++cov1ReductionStep) {
    unsigned int cov1Stride = cov1ReductionStride;
    if ((lid < cov1Stride)) {
      float otherCov = scov[(lid + cov1Stride)];
      float otherVX = svarX[(lid + cov1Stride)];
      float otherVY = svarY[(lid + cov1Stride)];
      unsigned int otherC = scount[(lid + cov1Stride)];
      float mineCov = scov[lid];
      float mineVX = svarX[lid];
      float mineVY = svarY[lid];
      unsigned int mineC = scount[lid];
      float sumCov = (mineCov + otherCov);
      float sumVX = (mineVX + otherVX);
      float sumVY = (mineVY + otherVY);
      unsigned int sumC = (mineC + otherC);
      scov[lid] = sumCov;
      svarX[lid] = sumVX;
      svarY[lid] = sumVY;
      scount[lid] = sumC;
    }
    unsigned int cov1StrideToHalve = cov1ReductionStride;
    unsigned int cov1NextStride = (cov1StrideToHalve >> 1u);
    cov1ReductionStride = cov1NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float totalCov = scov[0];
    float totalVX = svarX[0];
    float totalVY = svarY[0];
    unsigned int totalCount = scount[0];
    float countF = ((float)(totalCount));
    float safeCount = (((countF > __uint_as_float(0x00000000u))) ? (countF) : (__uint_as_float(0x3f800000u)));
    float covVal = (totalCov / safeCount);
    float varX = (totalVX / safeCount);
    float varY = (totalVY / safeCount);
    float covSq = (covVal * covVal);
    float det = ((varX * varY) - covSq);
    robustCov[0] = covVal;
    determinant[0] = det;
  }
}
