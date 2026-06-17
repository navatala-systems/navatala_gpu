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

kernel void navatala_cfd_vof_phir(device const float* gx [[buffer(0)]], device const float* gy [[buffer(1)]], device const float* gz [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* phi [[buffer(6)]], device const float* magSf [[buffer(7)]], device const int* owner [[buffer(8)]], device const int* nei [[buffer(9)]], device const int* faceParams [[buffer(10)]], device const float* paramsF [[buffer(11)]], device float* outPhir [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= faceParams[0])) {
    return;
  } else {
    if ((paramsF[0] == as_type<float>(0x00000000u))) {
      outPhir[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      return;
    }
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float gox = gx[o];
    float goy = gy[o];
    float goz = gz[o];
    float gnx = gx[n];
    float gny = gy[n];
    float gnz = gz[n];
    float gfx = (as_type<float>(0x3f000000u) * (gox + gnx));
    float gfy = (as_type<float>(0x3f000000u) * (goy + gny));
    float gfz = (as_type<float>(0x3f000000u) * (goz + gnz));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float dotSf = (((gfx * sfX[((int)(int(__gid.x)))]) + (gfy * sfY[((int)(int(__gid.x)))])) + (gfz * sfZ[((int)(int(__gid.x)))]));
    float nHatf = (dotSf / (magg + paramsF[1]));
    float phic = ((paramsF[0] * abs(phi[((int)(int(__gid.x)))])) / magSf[((int)(int(__gid.x)))]);
    outPhir[((int)(int(__gid.x)))] = (phic * nHatf);
  }
}
