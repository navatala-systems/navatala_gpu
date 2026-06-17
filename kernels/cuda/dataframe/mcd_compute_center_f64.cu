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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_mcd_compute_center_f64(const double* dataX, const double* dataY, const unsigned int* mask, const unsigned int* n, double* robustMeanX, double* robustMeanY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdataX[256];
  __shared__ double sdataY[256];
  __shared__ unsigned int scount[256];
  unsigned int len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    unsigned int m = mask[gid];
    double x = dataX[gid];
    double y = dataY[gid];
    double mF = ((double)(m));
    double maskedX = (x * mF);
    double maskedY = (y * mF);
    sdataX[lid] = maskedX;
    sdataY[lid] = maskedY;
    scount[lid] = m;
  } else {
    sdataX[lid] = __longlong_as_double(0x0000000000000000ull);
    sdataY[lid] = __longlong_as_double(0x0000000000000000ull);
    scount[lid] = 0u;
  }
  __syncthreads();
  unsigned int ctr2ReductionStride = 128u;
  for (int ctr2ReductionStep = 0; ctr2ReductionStep < (int)(8); ++ctr2ReductionStep) {
    unsigned int ctr2Stride = ctr2ReductionStride;
    if ((lid < ctr2Stride)) {
      double otherX = sdataX[(lid + ctr2Stride)];
      double otherY = sdataY[(lid + ctr2Stride)];
      unsigned int otherC = scount[(lid + ctr2Stride)];
      double mineX = sdataX[lid];
      double mineY = sdataY[lid];
      unsigned int mineC = scount[lid];
      double sumX = (mineX + otherX);
      double sumY = (mineY + otherY);
      unsigned int sumC = (mineC + otherC);
      sdataX[lid] = sumX;
      sdataY[lid] = sumY;
      scount[lid] = sumC;
    }
    unsigned int ctr2StrideToHalve = ctr2ReductionStride;
    unsigned int ctr2NextStride = (ctr2StrideToHalve >> 1u);
    ctr2ReductionStride = ctr2NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double totalX = sdataX[0];
    double totalY = sdataY[0];
    unsigned int totalCount = scount[0];
    double countF = ((double)(totalCount));
    double safeCount = (((countF > __longlong_as_double(0x0000000000000000ull))) ? (countF) : (__longlong_as_double(0x3ff0000000000000ull)));
    double meanX = (totalX / safeCount);
    double meanY = (totalY / safeCount);
    robustMeanX[0] = meanX;
    robustMeanY[0] = meanY;
  }
}
