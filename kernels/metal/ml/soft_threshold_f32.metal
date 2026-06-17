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

kernel void navatala_ml_soft_threshold_f32(device const float* value [[buffer(0)]], device const float* threshold [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  bool inBounds = (gid == 0u);
  if (inBounds) {
    float v = value[0];
    float t = threshold[0];
    float absV = abs(v);
    float diff = (absV - t);
    float maxDiff = (((diff > as_type<float>(0x00000000u))) ? (diff) : (as_type<float>(0x00000000u)));
    bool positive = (v > as_type<float>(0x00000000u));
    bool negative = (v < as_type<float>(0x00000000u));
    float signVal = ((positive) ? (as_type<float>(0x3f800000u)) : (((negative) ? (as_type<float>(0xbf800000u)) : (as_type<float>(0x00000000u)))));
    float res = (signVal * maxDiff);
    result[0] = res;
  }
}
