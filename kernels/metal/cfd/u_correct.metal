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

kernel void navatala_cfd_u_correct(device const float* pCell [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* bcVal [[buffer(6)]], device const int* bcMask [[buffer(7)]], device const int* offsets [[buffer(8)]], device const int* faceIdx [[buffer(9)]], device const float* sign [[buffer(10)]], device const float* vol [[buffer(11)]], device const float* rAU [[buffer(12)]], device const float* ux [[buffer(13)]], device const float* uy [[buffer(14)]], device const float* uz [[buffer(15)]], device const int* counts [[buffer(16)]], device const float* paramsF [[buffer(17)]], device float* outX [[buffer(18)]], device float* outY [[buffer(19)]], device float* outZ [[buffer(20)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = as_type<float>(0x00000000u);
    float gy = as_type<float>(0x00000000u);
    float gz = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if (f < counts[2]) {
        int n = neighbour[f];
        pf = (as_type<float>(0x3f000000u) * (pCell[o] + pCell[n]));
      } else {
        if (bcMask[f] != 0) {
          pf = bcVal[f];
        }
      }
      float s = sign[k];
      gx = (gx + (s * (sfX[f] * pf)));
      gy = (gy + (s * (sfY[f] * pf)));
      gz = (gz + (s * (sfZ[f] * pf)));
    }
    float invV = (as_type<float>(0x3f800000u) / vol[((int)(int(__gid.x)))]);
    float sC = (paramsF[0] * rAU[((int)(int(__gid.x)))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)(int(__gid.x)))] = (ux[((int)(int(__gid.x)))] - dUx);
    outY[((int)(int(__gid.x)))] = (uy[((int)(int(__gid.x)))] - dUy);
    outZ[((int)(int(__gid.x)))] = (uz[((int)(int(__gid.x)))] - dUz);
  }
}
