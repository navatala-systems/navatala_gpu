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
extern "C" __global__ void navatala_ml_compute_r_squared_f32(const float* y, const float* yPred, const float* yMean, const unsigned int* nSamples, float* rSquared) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdataRes[256];
  __shared__ float sdataTot[256];
  unsigned int n = nSamples[0];
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
    sdataRes[lid] = __uint_as_float(0x00000000u);
    sdataTot[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int r2F32Stride = 128u;
  for (int reductionStep3 = 0; reductionStep3 < (int)(8); ++reductionStep3) {
    unsigned int stride3 = r2F32Stride;
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
    unsigned int strideToHalve3 = r2F32Stride;
    unsigned int nextStride3 = (strideToHalve3 >> 1u);
    r2F32Stride = nextStride3;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float ssRes = sdataRes[0];
    float ssTot = sdataTot[0];
    float ratio = (ssRes / ssTot);
    float r2 = (__uint_as_float(0x3f800000u) - ratio);
    rSquared[0] = r2;
  }
}
