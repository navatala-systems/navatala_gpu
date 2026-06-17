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
extern "C" __global__ void navatala_cfd_laplacian_scalar_explicit(const float* muCell, const float* phiCell, const int* owner, const int* neighbour, const float* weights, const float* magSf, const float* deltaCoeffs, const int* offsets, const int* faceIdx, const int* sign, const float* vol, const float* bcMu, const unsigned int* bcMuMask, const float* bcPhi, const unsigned int* bcPhiMask, const int* params, float* outL) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0])) {
    return;
  } else {
    float phiC = phiCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float muC = muCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float sum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        int sgn = sign[k];
        int o = owner[f];
        float phiOther = phiCell[o];
        float muF = muC;
        if ((sgn > 0)) {
          if ((f < params[2])) {
            int n = neighbour[f];
            phiOther = phiCell[n];
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if ((bcPhiMask[f] != 0u)) {
              phiOther = bcPhi[f];
            }
            if ((bcMuMask[f] != 0u)) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if ((v != __uint_as_float(0x00000000u))) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outL[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sum * invV);
  }
}
