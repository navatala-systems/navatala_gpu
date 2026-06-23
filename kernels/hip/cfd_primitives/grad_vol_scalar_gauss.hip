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
extern "C" __global__ void navatala_cfd_primitives_grad_vol_scalar_gauss(const float* cellPhi, const float* bcValue, const unsigned int* bcMask, const int* owner, const int* neighbour, const float* weights, const float* sfX, const float* sfY, const float* sfZ, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float sumX = __uint_as_float(0x00000000u);
    float sumY = __uint_as_float(0x00000000u);
    float sumZ = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = sign[k];
        int o = owner[f];
        float po = cellPhi[o];
        float phiF = po;
        if (f < params[2]) {
          int n = neighbour[f];
          float pn = cellPhi[n];
          float ww = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          unsigned int m = bcMask[f];
          if (m == 1u) {
            float ww = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - ww);
            phiF = ((ww * po) + (iw * bcValue[f]));
          }
          if (m == 2u) {
            phiF = bcValue[f];
          }
        }
        float w = (s * phiF);
        sumX = (sumX + (w * sfX[f]));
        sumY = (sumY + (w * sfY[f]));
        sumZ = (sumZ + (w * sfZ[f]));
      }
    }
    float v = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float invV = __uint_as_float(0x00000000u);
    if (v != __uint_as_float(0x00000000u)) {
      invV = (__uint_as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumX * invV);
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumY * invV);
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (sumZ * invV);
  }
}
