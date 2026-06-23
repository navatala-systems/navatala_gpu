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

kernel void navatala_cfd_laplacian_scalar_explicit(device const float* muCell [[buffer(0)]], device const float* phiCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* weights [[buffer(4)]], device const float* magSf [[buffer(5)]], device const float* deltaCoeffs [[buffer(6)]], device const int* offsets [[buffer(7)]], device const int* faceIdx [[buffer(8)]], device const int* sign [[buffer(9)]], device const float* vol [[buffer(10)]], device const float* bcMu [[buffer(11)]], device const uint* bcMuMask [[buffer(12)]], device const float* bcPhi [[buffer(13)]], device const uint* bcPhiMask [[buffer(14)]], device const int* params [[buffer(15)]], device float* outL [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float phiC = phiCell[((int)(int(__gid.x)))];
    float muC = muCell[((int)(int(__gid.x)))];
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        int sgn = sign[k];
        int o = owner[f];
        float phiOther = phiCell[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            phiOther = phiCell[n];
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if (bcPhiMask[f] != 0u) {
              phiOther = bcPhi[f];
            }
            if (bcMuMask[f] != 0u) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_type<float>(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outL[((int)(int(__gid.x)))] = (sum * invV);
  }
}
