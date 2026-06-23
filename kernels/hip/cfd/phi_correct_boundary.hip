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
extern "C" __global__ void navatala_cfd_phi_correct_boundary(const float* phiIn, const float* pCell, const int* owner, const float* magSf, const float* deltaCoeffs, const float* coeff, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* phiOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int f = (counts[1] + ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))));
  if (f >= counts[0]) {
    return;
  } else {
    int o = owner[f];
    float po = pCell[o];
    float other = po;
    if (bcMask[f] != 0) {
      other = bcVal[f];
    }
    float snGrad = (deltaCoeffs[f] * (other - po));
    if (bcSnGradMask[f] != 0) {
      snGrad = bcSnGrad[f];
    }
    float gradFace = (magSf[f] * snGrad);
    phiOut[f] = (phiIn[f] - ((paramsF[0] * (coeff[f] * gradFace)) + faceFluxCorrection[f]));
  }
}
