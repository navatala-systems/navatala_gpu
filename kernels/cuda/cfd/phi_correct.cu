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
extern "C" __global__ void navatala_cfd_phi_correct(const float* phiIn, const float* pCell, const int* owner, const int* neighbour, const float* magSf, const float* deltaCoeffs, const float* coeff, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* phiOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float po = pCell[o];
    float other = po;
    float snGrad = __uint_as_float(0x00000000u);
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[1])) {
      int n = neighbour[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      other = pCell[n];
    } else {
      if ((bcMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0)) {
        other = bcVal[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      }
    }
    snGrad = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (other - po));
    if ((bcSnGradMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0)) {
      snGrad = bcSnGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    }
    float gradFace = (magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * snGrad);
    phiOut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (phiIn[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - ((paramsF[0] * (coeff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * gradFace)) + faceFluxCorrection[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
  }
}
