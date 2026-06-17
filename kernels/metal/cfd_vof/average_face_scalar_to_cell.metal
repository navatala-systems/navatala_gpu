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

kernel void navatala_cfd_vof_average_face_scalar_to_cell(device const float* alphaF [[buffer(0)]], device const float* magSf [[buffer(1)]], device const uint* offsets [[buffer(2)]], device const uint* faceIdx [[buffer(3)]], device const float* sumMag [[buffer(4)]], device const uint* counts [[buffer(5)]], device float* outAlpha [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(counts[0])))) {
    return;
  } else {
    float s = as_type<float>(0x00000000u);
    uint beg = offsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      s = (s + (magSf[((int)(f))] * alphaF[((int)(f))]));
    }
    float d = sumMag[int(__gid.x)];
    if ((d < as_type<float>(0x1e3ce508u))) {
      outAlpha[int(__gid.x)] = (s / as_type<float>(0x1e3ce508u));
    } else {
      outAlpha[int(__gid.x)] = (s / d);
    }
  }
}
