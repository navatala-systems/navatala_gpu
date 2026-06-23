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

kernel void navatala_cfd_primitives_grad_vol_scalar_gauss(device const float* cellPhi [[buffer(0)]], device const float* bcValue [[buffer(1)]], device const uint* bcMask [[buffer(2)]], device const int* owner [[buffer(3)]], device const int* neighbour [[buffer(4)]], device const float* weights [[buffer(5)]], device const float* sfX [[buffer(6)]], device const float* sfY [[buffer(7)]], device const float* sfZ [[buffer(8)]], device const int* offsets [[buffer(9)]], device const int* faceIdx [[buffer(10)]], device const float* sign [[buffer(11)]], device const float* vol [[buffer(12)]], device const int* params [[buffer(13)]], device float* outX [[buffer(14)]], device float* outY [[buffer(15)]], device float* outZ [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
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
        int o = owner[f];
        float po = cellPhi[o];
        float phiF = po;
        if (f < params[2]) {
          int n = neighbour[f];
          float pn = cellPhi[n];
          float ww = weights[f];
          float iw = (as_type<float>(0x3f800000u) - ww);
          phiF = ((ww * po) + (iw * pn));
        } else {
          uint m = bcMask[f];
          if (m == 1u) {
            float ww = weights[f];
            float iw = (as_type<float>(0x3f800000u) - ww);
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
