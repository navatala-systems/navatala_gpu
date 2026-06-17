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

kernel void navatala_dataframe_init_p2_state_f32(device const float* p [[buffer(0)]], device float* q0 [[buffer(1)]], device float* q1 [[buffer(2)]], device float* q2 [[buffer(3)]], device float* q3 [[buffer(4)]], device float* q4 [[buffer(5)]], device uint* pos0 [[buffer(6)]], device uint* pos1 [[buffer(7)]], device uint* pos2 [[buffer(8)]], device uint* pos3 [[buffer(9)]], device uint* pos4 [[buffer(10)]], device float* targetP [[buffer(11)]], device uint* count [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) == 0)) {
    float pVal = p[0];
    q0[0] = as_type<float>(0x00000000u);
    q1[0] = as_type<float>(0x00000000u);
    q2[0] = as_type<float>(0x00000000u);
    q3[0] = as_type<float>(0x00000000u);
    q4[0] = as_type<float>(0x00000000u);
    pos0[0] = 0u;
    pos1[0] = 0u;
    pos2[0] = 0u;
    pos3[0] = 0u;
    pos4[0] = 0u;
    targetP[0] = pVal;
    count[0] = 0u;
  }
}
