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

kernel void navatala_cfd_gamg_scale_correction(device const float* z [[buffer(0)]], device const float* r [[buffer(1)]], device const float* Az [[buffer(2)]], device const float* diagInv [[buffer(3)]], device const float* sfPtr [[buffer(4)]], device const uint* nRows [[buffer(5)]], device float* corr [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) >= ((int)(nRows[0])))) {
    return;
  } else {
    float sf = sfPtr[0];
    float zi = z[int(__gid.x)];
    float ri = r[int(__gid.x)];
    float azi = Az[int(__gid.x)];
    float di = diagInv[int(__gid.x)];
    float sfz = (sf * zi);
    float sfaz = (sf * azi);
    float residual = (ri - sfaz);
    float corrVal = (sfz + (residual * di));
    corr[int(__gid.x)] = corrVal;
  }
}
