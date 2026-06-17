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
extern "C" __global__ void navatala_cfd_stf_all_faces(const float* divN, const float* alpha, const float* alphaF, const float* delta, const int* owner, const int* nei, const float* weights, const float* contactGrad, const unsigned int* thetaMask, const float* paramsF, const int* counts, float* stf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float w = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float wm = (__uint_as_float(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (__uint_as_float(0xbf800000u) * divInterp);
    float snGrad = __uint_as_float(0x00000000u);
    if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[2])) {
      snGrad = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (alpha[n] - alpha[o]));
    } else {
      if ((thetaMask[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] != 0u)) {
        snGrad = contactGrad[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
      } else {
        snGrad = (delta[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] - alpha[o]));
      }
    }
    stf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((paramsF[0] * Kf) * snGrad);
  }
}
