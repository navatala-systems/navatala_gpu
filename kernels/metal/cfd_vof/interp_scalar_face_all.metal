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

kernel void navatala_cfd_vof_interp_scalar_face_all(device const float* alpha [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* nei [[buffer(2)]], device const float* w [[buffer(3)]], device const float* bcAlphaF [[buffer(4)]], device const uint* params [[buffer(5)]], device float* alphaF [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(params[0])))) {
    return;
  } else {
    if ((int(__gid.x) < ((int)(params[1])))) {
      int o = owner[int(__gid.x)];
      int n = nei[int(__gid.x)];
      float wf = w[int(__gid.x)];
      float wc = (wf * ((float)((wf > as_type<float>(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((as_type<float>(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > as_type<float>(0x00000000u)))));
      if ((ac > as_type<float>(0x3f800000u))) {
        alphaF[int(__gid.x)] = as_type<float>(0x3f800000u);
      } else {
        alphaF[int(__gid.x)] = ac;
      }
    } else {
      float a = bcAlphaF[int(__gid.x)];
      float ac = (a * ((float)((a > as_type<float>(0x00000000u)))));
      if ((ac > as_type<float>(0x3f800000u))) {
        alphaF[int(__gid.x)] = as_type<float>(0x3f800000u);
      } else {
        alphaF[int(__gid.x)] = ac;
      }
    }
  }
}
