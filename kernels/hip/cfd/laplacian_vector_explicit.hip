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
extern "C" __global__ void navatala_cfd_laplacian_vector_explicit(const float* muCell, const float* muBc, const float* ux, const float* uy, const float* uz, const int* owner, const int* neighbour, const float* weights, const float* magSf, const float* deltaCoeffs, const float* corrX, const float* corrY, const float* corrZ, const float* gradRow0, const float* gradRow1, const float* gradRow2, const int* offsets, const int* faceIdx, const int* sign, const float* vol, const float* bcX, const float* bcY, const float* bcZ, const int* bcMask, const int* params, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= params[0]) {
    return;
  } else {
    float ucx = ux[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ucy = uy[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float ucz = uz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float muC = muCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
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
        int sgn = sign[k];
        int o = owner[f];
        float otherX = ux[o];
        float otherY = uy[o];
        float otherZ = uz[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            otherX = ux[n];
            otherY = uy[n];
            otherZ = uz[n];
            float w0 = weights[f];
            float iw = (__uint_as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            muF = muBc[f];
            if (bcMask[f] == 1) {
              float w0 = weights[f];
              float iw = (__uint_as_float(0x3f800000u) - w0);
              muF = ((w0 * muC) + (iw * muBc[f]));
              otherX = bcX[f];
              otherY = bcY[f];
              otherZ = bcZ[f];
            } else {
              if (bcMask[f] != 0) {
                otherX = bcX[f];
                otherY = bcY[f];
                otherZ = bcZ[f];
              }
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (__uint_as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sumX = (sumX + (coeff * (otherX - ucx)));
        sumY = (sumY + (coeff * (otherY - ucy)));
        sumZ = (sumZ + (coeff * (otherZ - ucz)));
        float kx = corrX[f];
        float ky = corrY[f];
        float kz = corrZ[f];
        float gxxF = gradRow0[((o)*3 + 0)];
        float gxyF = gradRow0[((o)*3 + 1)];
        float gxzF = gradRow0[((o)*3 + 2)];
        float gyxF = gradRow1[((o)*3 + 0)];
        float gyyF = gradRow1[((o)*3 + 1)];
        float gyzF = gradRow1[((o)*3 + 2)];
        float gzxF = gradRow2[((o)*3 + 0)];
        float gzyF = gradRow2[((o)*3 + 1)];
        float gzzF = gradRow2[((o)*3 + 2)];
        if (f < params[2]) {
          int n2 = neighbour[f];
          float w1 = weights[f];
          float iw1 = (__uint_as_float(0x3f800000u) - w1);
          gxxF = ((w1 * gradRow0[((o)*3 + 0)]) + (iw1 * gradRow0[((n2)*3 + 0)]));
          gxyF = ((w1 * gradRow0[((o)*3 + 1)]) + (iw1 * gradRow0[((n2)*3 + 1)]));
          gxzF = ((w1 * gradRow0[((o)*3 + 2)]) + (iw1 * gradRow0[((n2)*3 + 2)]));
          gyxF = ((w1 * gradRow1[((o)*3 + 0)]) + (iw1 * gradRow1[((n2)*3 + 0)]));
          gyyF = ((w1 * gradRow1[((o)*3 + 1)]) + (iw1 * gradRow1[((n2)*3 + 1)]));
          gyzF = ((w1 * gradRow1[((o)*3 + 2)]) + (iw1 * gradRow1[((n2)*3 + 2)]));
          gzxF = ((w1 * gradRow2[((o)*3 + 0)]) + (iw1 * gradRow2[((n2)*3 + 0)]));
          gzyF = ((w1 * gradRow2[((o)*3 + 1)]) + (iw1 * gradRow2[((n2)*3 + 1)]));
          gzzF = ((w1 * gradRow2[((o)*3 + 2)]) + (iw1 * gradRow2[((n2)*3 + 2)]));
        }
        float corrUx = (((kx * gxxF) + (ky * gxyF)) + (kz * gxzF));
        float corrUy = (((kx * gyxF) + (ky * gyyF)) + (kz * gyzF));
        float corrUz = (((kx * gzxF) + (ky * gzyF)) + (kz * gzzF));
        float fluxScale = (muF * magSf[f]);
        float sgnF = ((float)(sgn));
        sumX = (sumX + ((sgnF * fluxScale) * corrUx));
        sumY = (sumY + ((sgnF * fluxScale) * corrUy));
        sumZ = (sumZ + ((sgnF * fluxScale) * corrUz));
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
