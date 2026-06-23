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

kernel void navatala_cfd_div_upwind_coeffs(device const float* weights [[buffer(0)]], device const float* flux [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outLower [[buffer(3)]], device float* outUpper [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float w = weights[((int)(int(__gid.x)))];
    float phi = flux[((int)(int(__gid.x)))];
    float low = ((as_type<float>(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)(int(__gid.x)))] = low;
    outUpper[((int)(int(__gid.x)))] = up;
  }
}
