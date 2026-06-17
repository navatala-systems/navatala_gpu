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

kernel void navatala_ml_bernoulli_f32(device const uint* count [[buffer(0)]], device const uint* seed [[buffer(1)]], device const float* p [[buffer(2)]], device float* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  uint seedVal = seed[0];
  if ((gid < countVal)) {
    uint uk0 = ((seedVal + (gid * 2654435769u)) + 0u);
    uint uk1 = ((uk0 ^ (uk0 >> 16u)) * 2246822519u);
    uint uk2 = ((uk1 ^ (uk1 >> 13u)) * 3266489917u);
    uint uk3 = (uk2 ^ (uk2 >> 16u));
    float uF = (((float)(uk3)) * as_type<float>(0x2f800000u));
    float pVal = p[0];
    float bit = (((uF < pVal)) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
    float outV = bit;
    _output[gid] = outV;
  }
}
