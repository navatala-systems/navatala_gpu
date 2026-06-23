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

kernel void navatala_cfd_hby_a_from_u_and_terms(device const float* ux [[buffer(0)]], device const float* uy [[buffer(1)]], device const float* uz [[buffer(2)]], device const float* lapx [[buffer(3)]], device const float* lapy [[buffer(4)]], device const float* lapz [[buffer(5)]], device const float* divx [[buffer(6)]], device const float* divy [[buffer(7)]], device const float* divz [[buffer(8)]], device const float* extrax [[buffer(9)]], device const float* extray [[buffer(10)]], device const float* extraz [[buffer(11)]], device const float* rau [[buffer(12)]], device const int* counts [[buffer(13)]], device const int* modeBuf [[buffer(14)]], device float* outx [[buffer(15)]], device float* outy [[buffer(16)]], device float* outz [[buffer(17)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float tx = lapx[((int)(int(__gid.x)))];
    float ty = lapy[((int)(int(__gid.x)))];
    float tz = lapz[((int)(int(__gid.x)))];
    if ((modeBuf[0] == 1) || (modeBuf[0] == 3)) {
      tx = (tx - divx[((int)(int(__gid.x)))]);
      ty = (ty - divy[((int)(int(__gid.x)))]);
      tz = (tz - divz[((int)(int(__gid.x)))]);
    }
    if ((modeBuf[0] == 2) || (modeBuf[0] == 3)) {
      tx = (tx + extrax[((int)(int(__gid.x)))]);
      ty = (ty + extray[((int)(int(__gid.x)))]);
      tz = (tz + extraz[((int)(int(__gid.x)))]);
    }
    float a = rau[((int)(int(__gid.x)))];
    outx[((int)(int(__gid.x)))] = (ux[((int)(int(__gid.x)))] + (a * tx));
    outy[((int)(int(__gid.x)))] = (uy[((int)(int(__gid.x)))] + (a * ty));
    outz[((int)(int(__gid.x)))] = (uz[((int)(int(__gid.x)))] + (a * tz));
  }
}
