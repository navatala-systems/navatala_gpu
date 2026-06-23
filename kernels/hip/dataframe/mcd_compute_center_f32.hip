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
extern "C" __global__ void navatala_dataframe_mcd_compute_center_f32(const float* dataX, const float* dataY, const unsigned int* mask, const unsigned int* n, float* robustMeanX, float* robustMeanY) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdataX[256];
  __shared__ float sdataY[256];
  __shared__ unsigned int scount[256];
  unsigned int len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    unsigned int m = mask[gid];
    float x = dataX[gid];
    float y = dataY[gid];
    float mF = ((float)(m));
    float maskedX = (x * mF);
    float maskedY = (y * mF);
    sdataX[lid] = maskedX;
    sdataY[lid] = maskedY;
    scount[lid] = m;
  } else {
    sdataX[lid] = __uint_as_float(0x00000000u);
    sdataY[lid] = __uint_as_float(0x00000000u);
    scount[lid] = 0u;
  }
  __syncthreads();
  unsigned int ctr1ReductionStride = 128u;
  for (int ctr1ReductionStep = 0; ctr1ReductionStep < (int)(8); ++ctr1ReductionStep) {
    unsigned int ctr1Stride = ctr1ReductionStride;
    if (lid < ctr1Stride) {
      float otherX = sdataX[(lid + ctr1Stride)];
      float otherY = sdataY[(lid + ctr1Stride)];
      unsigned int otherC = scount[(lid + ctr1Stride)];
      float mineX = sdataX[lid];
      float mineY = sdataY[lid];
      unsigned int mineC = scount[lid];
      float sumX = (mineX + otherX);
      float sumY = (mineY + otherY);
      unsigned int sumC = (mineC + otherC);
      sdataX[lid] = sumX;
      sdataY[lid] = sumY;
      scount[lid] = sumC;
    }
    unsigned int ctr1StrideToHalve = ctr1ReductionStride;
    unsigned int ctr1NextStride = (ctr1StrideToHalve >> 1u);
    ctr1ReductionStride = ctr1NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float totalX = sdataX[0];
    float totalY = sdataY[0];
    unsigned int totalCount = scount[0];
    float countF = ((float)(totalCount));
    float safeCount = (((countF > __uint_as_float(0x00000000u))) ? (countF) : (__uint_as_float(0x3f800000u)));
    float meanX = (totalX / safeCount);
    float meanY = (totalY / safeCount);
    robustMeanX[0] = meanX;
    robustMeanY[0] = meanY;
  }
}
