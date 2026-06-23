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

kernel void navatala_cfd_primitives_phi_from_u_internal(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* ux [[buffer(6)]], device const float* uy [[buffer(7)]], device const float* uz [[buffer(8)]], device const uint* params [[buffer(9)]], device float* outPhi [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[1]))) {
    return;
  } else {
    int o = owner[int(__gid.x)];
    int n = neighbour[int(__gid.x)];
    float w = weights[int(__gid.x)];
    float iw = (as_type<float>(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[int(__gid.x)] = (((sfX[int(__gid.x)] * ufx) + (sfY[int(__gid.x)] * ufy)) + (sfZ[int(__gid.x)] * ufz));
  }
}
