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

kernel void navatala_cfd_primitives_linear_upwind_scalar_face(device const float* flux [[buffer(0)]], device const float* xCell [[buffer(1)]], device const float* gradX [[buffer(2)]], device const float* gradY [[buffer(3)]], device const float* gradZ [[buffer(4)]], device const float* limiter [[buffer(5)]], device const int* owner [[buffer(6)]], device const int* nei [[buffer(7)]], device const float* cx [[buffer(8)]], device const float* cy [[buffer(9)]], device const float* cz [[buffer(10)]], device const float* cfx [[buffer(11)]], device const float* cfy [[buffer(12)]], device const float* cfz [[buffer(13)]], device const float* bcValue [[buffer(14)]], device const uint* bcMask [[buffer(15)]], device const uint* params [[buffer(16)]], device float* outFace [[buffer(17)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(params[0])))) {
    return;
  } else {
    float phif = flux[int(__gid.x)];
    int own = owner[int(__gid.x)];
    if ((int(__gid.x) < ((int)(params[1])))) {
      if ((phif >= as_type<float>(0x00000000u))) {
        float dx = (cfx[int(__gid.x)] - cx[own]);
        float dy = (cfy[int(__gid.x)] - cy[own]);
        float dz = (cfz[int(__gid.x)] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[int(__gid.x)] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[int(__gid.x)];
        float dx = (cfx[int(__gid.x)] - cx[neiCell]);
        float dy = (cfy[int(__gid.x)] - cy[neiCell]);
        float dz = (cfz[int(__gid.x)] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[int(__gid.x)] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if ((phif >= as_type<float>(0x00000000u))) {
        outFace[int(__gid.x)] = xCell[own];
      } else {
        uint mask = bcMask[int(__gid.x)];
        if ((((int)(mask)) == 0)) {
          outFace[int(__gid.x)] = xCell[own];
        } else {
          outFace[int(__gid.x)] = bcValue[int(__gid.x)];
        }
      }
    }
  }
}
