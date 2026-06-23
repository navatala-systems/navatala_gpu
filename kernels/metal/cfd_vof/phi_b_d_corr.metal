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

kernel void navatala_cfd_vof_phi_b_d_corr(device const float* alpha [[buffer(0)]], device const float* phiAll [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device const float* alphaF [[buffer(4)]], device const float* alphaPhiAll [[buffer(5)]], device const int* params [[buffer(6)]], device float* phiBD [[buffer(7)]], device float* phiCorr [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    if (((int)(int(__gid.x))) < params[1]) {
      float ph = phiAll[((int)(int(__gid.x)))];
      int o = owner[((int)(int(__gid.x)))];
      int n = nei[((int)(int(__gid.x)))];
      float a = as_type<float>(0x00000000u);
      if (ph >= as_type<float>(0x00000000u)) {
        a = alpha[o];
      } else {
        a = alpha[n];
      }
      if (a < as_type<float>(0x00000000u)) {
        a = as_type<float>(0x00000000u);
      }
      if (a > as_type<float>(0x3f800000u)) {
        a = as_type<float>(0x3f800000u);
      }
      float pbd = (ph * a);
      phiBD[((int)(int(__gid.x)))] = pbd;
      phiCorr[((int)(int(__gid.x)))] = (alphaPhiAll[((int)(int(__gid.x)))] - pbd);
    } else {
      phiBD[((int)(int(__gid.x)))] = alphaPhiAll[((int)(int(__gid.x)))];
      phiCorr[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}
