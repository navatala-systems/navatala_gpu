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

kernel void navatala_cfd_grad_alpha_cell(device const float* alphaF [[buffer(0)]], device const float* sfX [[buffer(1)]], device const float* sfY [[buffer(2)]], device const float* sfZ [[buffer(3)]], device const int* offsets [[buffer(4)]], device const int* faceIdx [[buffer(5)]], device const float* sign [[buffer(6)]], device const float* vol [[buffer(7)]], device const int* counts [[buffer(8)]], device float* gx [[buffer(9)]], device float* gy [[buffer(10)]], device float* gz [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sx = as_type<float>(0x00000000u);
    float sy = as_type<float>(0x00000000u);
    float sz = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = (sign[k] * alphaF[f]);
      sx = (sx + (s * sfX[f]));
      sy = (sy + (s * sfY[f]));
      sz = (sz + (s * sfZ[f]));
    }
    float invV = (as_type<float>(0x3f800000u) / vol[((int)(int(__gid.x)))]);
    gx[((int)(int(__gid.x)))] = (sx * invV);
    gy[((int)(int(__gid.x)))] = (sy * invV);
    gz[((int)(int(__gid.x)))] = (sz * invV);
  }
}
