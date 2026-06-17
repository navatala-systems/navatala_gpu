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

kernel void navatala_cfd_two_mag_sqr_symm_grad_u(device const float* ufX [[buffer(0)]], device const float* ufY [[buffer(1)]], device const float* ufZ [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const int* offsets [[buffer(6)]], device const int* faceIdx [[buffer(7)]], device const float* sign [[buffer(8)]], device const float* vol [[buffer(9)]], device const int* counts [[buffer(10)]], device float* outS2 [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float g00 = as_type<float>(0x00000000u);
    float g01 = as_type<float>(0x00000000u);
    float g02 = as_type<float>(0x00000000u);
    float g10 = as_type<float>(0x00000000u);
    float g11 = as_type<float>(0x00000000u);
    float g12 = as_type<float>(0x00000000u);
    float g20 = as_type<float>(0x00000000u);
    float g21 = as_type<float>(0x00000000u);
    float g22 = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float sx = (s * sfX[f]);
      float sy = (s * sfY[f]);
      float sz = (s * sfZ[f]);
      float ux = ufX[f];
      float uy = ufY[f];
      float uz = ufZ[f];
      g00 = (g00 + (sx * ux));
      g01 = (g01 + (sx * uy));
      g02 = (g02 + (sx * uz));
      g10 = (g10 + (sy * ux));
      g11 = (g11 + (sy * uy));
      g12 = (g12 + (sy * uz));
      g20 = (g20 + (sz * ux));
      g21 = (g21 + (sz * uy));
      g22 = (g22 + (sz * uz));
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if ((v != as_type<float>(0x00000000u))) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    g00 = (g00 * invV);
    g01 = (g01 * invV);
    g02 = (g02 * invV);
    g10 = (g10 * invV);
    g11 = (g11 * invV);
    g12 = (g12 * invV);
    g20 = (g20 * invV);
    g21 = (g21 * invV);
    g22 = (g22 * invV);
    float s00 = g00;
    float s11 = g11;
    float s22 = g22;
    float s01 = (as_type<float>(0x3f000000u) * (g01 + g10));
    float s02 = (as_type<float>(0x3f000000u) * (g02 + g20));
    float s12 = (as_type<float>(0x3f000000u) * (g12 + g21));
    float d0 = (s00 * s00);
    float d1 = (s11 * s11);
    float d2 = (s22 * s22);
    float o0 = (s01 * s01);
    float o1 = (s02 * s02);
    float o2 = (s12 * s12);
    float diag = ((d0 + d1) + d2);
    float off = ((o0 + o1) + o2);
    outS2[((int)(int(__gid.x)))] = ((as_type<float>(0x40000000u) * diag) + (as_type<float>(0x40800000u) * off));
  }
}
