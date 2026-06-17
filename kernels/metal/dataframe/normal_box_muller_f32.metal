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

kernel void navatala_dataframe_normal_box_muller_f32(device ulong* state [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0u];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    ulong s = state[gid];
    ulong aU64 = ((ulong)(1103515245u));
    ulong cU64 = ((ulong)(12345u));
    ulong s1 = (((aU64 * s) + cU64) & ((ulong)(2147483647u)));
    float u1 = (((float)(s1)) / as_type<float>(0x4f000000u));
    ulong s2 = (((aU64 * s1) + cU64) & ((ulong)(2147483647u)));
    float u2 = (((float)(s2)) / as_type<float>(0x4f000000u));
    state[gid] = s2;
    float u1Clamped = (((u1 < as_type<float>(0x33d6bf95u))) ? (as_type<float>(0x33d6bf95u)) : (u1));
    float logU1 = log(u1Clamped);
    float negTwoLogU1 = (as_type<float>(0xc0000000u) * logU1);
    float sqrtTerm = sqrt(negTwoLogU1);
    float angle = (as_type<float>(0x40c90fdbu) * u2);
    float cosTerm = cos(angle);
    float z = (sqrtTerm * cosTerm);
    _output[gid] = z;
  }
}
