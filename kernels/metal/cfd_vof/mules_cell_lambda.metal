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

kernel void navatala_cfd_vof_mules_cell_lambda(device const float* psiMaxCap [[buffer(0)]], device const float* psiMinCap [[buffer(1)]], device const float* sumPhip [[buffer(2)]], device const float* mSumPhim [[buffer(3)]], device const float* sumlPhip [[buffer(4)]], device const float* mSumlPhim [[buffer(5)]], device const int* counts [[buffer(6)]], device const float* paramsF [[buffer(7)]], device float* lambdam [[buffer(8)]], device float* lambdap [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= counts[0])) {
    return;
  } else {
    float ld = ((sumlPhip[((int)(int(__gid.x)))] + psiMaxCap[((int)(int(__gid.x)))]) / (mSumPhim[((int)(int(__gid.x)))] + paramsF[1]));
    float lp = ((mSumlPhim[((int)(int(__gid.x)))] + psiMinCap[((int)(int(__gid.x)))]) / (sumPhip[((int)(int(__gid.x)))] + paramsF[1]));
    if ((ld < as_type<float>(0x00000000u))) {
      ld = as_type<float>(0x00000000u);
    }
    if ((ld > as_type<float>(0x3f800000u))) {
      ld = as_type<float>(0x3f800000u);
    }
    if ((lp < as_type<float>(0x00000000u))) {
      lp = as_type<float>(0x00000000u);
    }
    if ((lp > as_type<float>(0x3f800000u))) {
      lp = as_type<float>(0x3f800000u);
    }
    lambdam[((int)(int(__gid.x)))] = ld;
    lambdap[((int)(int(__gid.x)))] = lp;
  }
}
