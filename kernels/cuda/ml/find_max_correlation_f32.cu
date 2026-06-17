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
extern "C" __global__ void navatala_ml_find_max_correlation_f32(const float* correlations, const unsigned int* activeMask, const unsigned int* nCols, unsigned int* maxIndex, float* maxValue) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float svals[256];
  __shared__ unsigned int sidxs[256];
  float otherVal = __uint_as_float(0x00000000u);
  unsigned int otherIdx = 0u;
  float myVal = __uint_as_float(0x00000000u);
  unsigned int myIdx = 0u;
  bool takeOther = false;
  unsigned int d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    unsigned int mask = activeMask[gid];
    bool isActive = (mask == 1u);
    float corr = correlations[gid];
    float absCorr = abs(corr);
    float val = ((isActive) ? (absCorr) : (__uint_as_float(0xff7fffffu)));
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = __uint_as_float(0xff7fffffu);
    sidxs[lid] = 0u;
  }
  __syncthreads();
  unsigned int maxCorrF32RedStride = 128u;
  for (int maxCorrF32RedStep = 0; maxCorrF32RedStep < (int)(8); ++maxCorrF32RedStep) {
    unsigned int maxCorrF32Stride = maxCorrF32RedStride;
    if ((lid < maxCorrF32Stride)) {
      otherVal = svals[(lid + maxCorrF32Stride)];
      otherIdx = sidxs[(lid + maxCorrF32Stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      takeOther = (otherVal > myVal);
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    unsigned int maxCorrF32StrideToHalve = maxCorrF32RedStride;
    unsigned int maxCorrF32NextStride = (maxCorrF32StrideToHalve >> 1u);
    maxCorrF32RedStride = maxCorrF32NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    maxValue[0] = svals[0];
    maxIndex[0] = sidxs[0];
  }
}
