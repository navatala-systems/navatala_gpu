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

kernel void navatala_dataframe_mad_f32(device const float* sortedData [[buffer(0)]], device const uint* n [[buffer(1)]], device const float* median [[buffer(2)]], device float* mad [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    uint len = n[0];
    float med = median[0];
    uint q1Idx = (len / 4u);
    uint q3Idx = (len - q1Idx);
    float q1Val = sortedData[q1Idx];
    float q3Val = sortedData[q3Idx];
    float dev1 = abs((q1Val - med));
    float dev3 = abs((q3Val - med));
    float madVal = ((dev1 + dev3) / as_type<float>(0x40000000u));
    mad[0] = madVal;
  }
}
