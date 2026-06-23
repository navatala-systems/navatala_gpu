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

kernel void navatala_cfd_div_rho_phi_u_explicit(device const float* phi [[buffer(0)]], device const float* rhoCell [[buffer(1)]], device const float* ux [[buffer(2)]], device const float* uy [[buffer(3)]], device const float* uz [[buffer(4)]], device const int* owner [[buffer(5)]], device const int* neighbour [[buffer(6)]], device const float* weights [[buffer(7)]], device const int* offsets [[buffer(8)]], device const int* faceIdx [[buffer(9)]], device const float* sign [[buffer(10)]], device const float* vol [[buffer(11)]], device const float* rhoBcVal [[buffer(12)]], device const uint* rhoBcMask [[buffer(13)]], device const float* uBcX [[buffer(14)]], device const float* uBcY [[buffer(15)]], device const float* uBcZ [[buffer(16)]], device const uint* uBcMask [[buffer(17)]], device const int* params [[buffer(18)]], device float* outX [[buffer(19)]], device float* outY [[buffer(20)]], device float* outZ [[buffer(21)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
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
      if (f < params[1]) {
        float s = sign[k];
        float flux = phi[f];
        int o = owner[f];
        float rhoF = rhoCell[o];
        float ufx = ux[o];
        float ufy = uy[o];
        float ufz = uz[o];
        if (f < params[2]) {
          int n = neighbour[f];
          float w0 = weights[f];
          float iw = (as_type<float>(0x3f800000u) - w0);
          rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
          ufx = ((w0 * ux[o]) + (iw * ux[n]));
          ufy = ((w0 * uy[o]) + (iw * uy[n]));
          ufz = ((w0 * uz[o]) + (iw * uz[n]));
        } else {
          uint rm = rhoBcMask[f];
          if (rm == 1u) {
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[f]));
          }
          if (rm == 2u) {
            rhoF = rhoBcVal[f];
          }
          uint um = uBcMask[f];
          if (um == 1u) {
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            ufx = ((w0 * ux[o]) + (iw * uBcX[f]));
            ufy = ((w0 * uy[o]) + (iw * uBcY[f]));
            ufz = ((w0 * uz[o]) + (iw * uBcZ[f]));
          }
          if (um == 2u) {
            ufx = uBcX[f];
            ufy = uBcY[f];
            ufz = uBcZ[f];
          }
        }
        float rhoPhi = (rhoF * flux);
        sumX = (sumX + (s * (rhoPhi * ufx)));
        sumY = (sumY + (s * (rhoPhi * ufy)));
        sumZ = (sumZ + (s * (rhoPhi * ufz)));
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}
