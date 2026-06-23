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
extern "C" __global__ void navatala_ml_find_max_correlation_f64(const double* correlations, const unsigned int* activeMask, const unsigned int* nCols, unsigned int* maxIndex, double* maxValue) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double svals[256];
  __shared__ unsigned int sidxs[256];
  double otherVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int otherIdx = 0u;
  double myVal = __longlong_as_double(0x0000000000000000ull);
  unsigned int myIdx = 0u;
  bool takeOther = false;
  unsigned int d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    unsigned int mask = activeMask[gid];
    bool isActive = (mask == 1u);
    double corr = correlations[gid];
    double absCorr = abs(corr);
    double val = ((isActive) ? (absCorr) : (__longlong_as_double(0xffefffffffffffffull)));
    svals[lid] = val;
    sidxs[lid] = gid;
  } else {
    svals[lid] = __longlong_as_double(0xffefffffffffffffull);
    sidxs[lid] = 0u;
  }
  __syncthreads();
  unsigned int maxCorrF64RedStride = 128u;
  for (int maxCorrF64RedStep = 0; maxCorrF64RedStep < (int)(8); ++maxCorrF64RedStep) {
    unsigned int maxCorrF64Stride = maxCorrF64RedStride;
    if (lid < maxCorrF64Stride) {
      otherVal = svals[(lid + maxCorrF64Stride)];
      otherIdx = sidxs[(lid + maxCorrF64Stride)];
      myVal = svals[lid];
      myIdx = sidxs[lid];
      takeOther = (otherVal > myVal);
      if (takeOther) {
        svals[lid] = otherVal;
        sidxs[lid] = otherIdx;
      }
    }
    unsigned int maxCorrF64StrideToHalve = maxCorrF64RedStride;
    unsigned int maxCorrF64NextStride = (maxCorrF64StrideToHalve >> 1u);
    maxCorrF64RedStride = maxCorrF64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    maxValue[0] = svals[0];
    maxIndex[0] = sidxs[0];
  }
}
