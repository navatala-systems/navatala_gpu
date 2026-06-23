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

kernel void navatala_dataframe_init_e_w_m_c_f32(device const float* alphaIn [[buffer(0)]], device float* alpha [[buffer(1)]], device float* nEff [[buffer(2)]], device float* meanX [[buffer(3)]], device float* meanY [[buffer(4)]], device float* Cxy [[buffer(5)]], device float* varX [[buffer(6)]], device float* varY [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float a = alphaIn[0];
    alpha[0] = a;
    nEff[0] = as_type<float>(0x00000000u);
    meanX[0] = as_type<float>(0x00000000u);
    meanY[0] = as_type<float>(0x00000000u);
    Cxy[0] = as_type<float>(0x00000000u);
    varX[0] = as_type<float>(0x00000000u);
    varY[0] = as_type<float>(0x00000000u);
  }
}
