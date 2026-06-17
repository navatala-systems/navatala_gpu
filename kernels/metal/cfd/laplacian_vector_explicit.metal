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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_vector_explicit(device const float* muCell [[buffer(0)]], device const float* muBc [[buffer(1)]], device const float* ux [[buffer(2)]], device const float* uy [[buffer(3)]], device const float* uz [[buffer(4)]], device const int* owner [[buffer(5)]], device const int* neighbour [[buffer(6)]], device const float* weights [[buffer(7)]], device const float* magSf [[buffer(8)]], device const float* deltaCoeffs [[buffer(9)]], device const float* corrX [[buffer(10)]], device const float* corrY [[buffer(11)]], device const float* corrZ [[buffer(12)]], device const float* gradRow0 [[buffer(13)]], device const float* gradRow1 [[buffer(14)]], device const float* gradRow2 [[buffer(15)]], device const int* offsets [[buffer(16)]], device const int* faceIdx [[buffer(17)]], device const int* sign [[buffer(18)]], device const float* vol [[buffer(19)]], device const float* bcX [[buffer(20)]], device const float* bcY [[buffer(21)]], device const float* bcZ [[buffer(22)]], device const int* bcMask [[buffer(23)]], device const int* params [[buffer(24)]], device float* outX [[buffer(25)]], device float* outY [[buffer(26)]], device float* outZ [[buffer(27)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= params[0])) {
    return;
  } else {
    float ucx = ux[((int)(int(__gid.x)))];
    float ucy = uy[((int)(int(__gid.x)))];
    float ucz = uz[((int)(int(__gid.x)))];
    float muC = muCell[((int)(int(__gid.x)))];
    float sumX = as_type<float>(0x00000000u);
    float sumY = as_type<float>(0x00000000u);
    float sumZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        int sgn = sign[k];
        int o = owner[f];
        float otherX = ux[o];
        float otherY = uy[o];
        float otherZ = uz[o];
        float muF = muC;
        if ((sgn > 0)) {
          if ((f < params[2])) {
            int n = neighbour[f];
            otherX = ux[n];
            otherY = uy[n];
            otherZ = uz[n];
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            muF = muBc[f];
            if ((bcMask[f] == 1)) {
              float w0 = weights[f];
              float iw = (as_type<float>(0x3f800000u) - w0);
              muF = ((w0 * muC) + (iw * muBc[f]));
              otherX = bcX[f];
              otherY = bcY[f];
              otherZ = bcZ[f];
            } else {
              if ((bcMask[f] != 0)) {
                otherX = bcX[f];
                otherY = bcY[f];
                otherZ = bcZ[f];
              }
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_type<float>(0x3f800000u) - w0);
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
        if ((f < params[2])) {
          int n2 = neighbour[f];
          float w1 = weights[f];
          float iw1 = (as_type<float>(0x3f800000u) - w1);
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
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if ((v != as_type<float>(0x00000000u))) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}
