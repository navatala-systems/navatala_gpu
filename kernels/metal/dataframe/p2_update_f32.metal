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

kernel void navatala_dataframe_p2_update_f32(device const float* x [[buffer(0)]], device float* q0 [[buffer(1)]], device float* q1 [[buffer(2)]], device float* q2 [[buffer(3)]], device float* q3 [[buffer(4)]], device float* q4 [[buffer(5)]], device uint* pos0 [[buffer(6)]], device uint* pos1 [[buffer(7)]], device uint* pos2 [[buffer(8)]], device uint* pos3 [[buffer(9)]], device uint* pos4 [[buffer(10)]], device const float* targetP [[buffer(11)]], device uint* count [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float xVal = x[0];
    uint n = count[0];
    float pVal = targetP[0];
    float h0 = q0[0];
    float h1 = q1[0];
    float h2 = q2[0];
    float h3 = q3[0];
    float h4 = q4[0];
    float newH0 = (((n == 0u)) ? (xVal) : ((((xVal < h0)) ? (xVal) : (h0))));
    float finalH0 = (((n >= 5u)) ? ((((xVal < h0)) ? (xVal) : (h0))) : (newH0));
    float finalH4 = (((n >= 5u)) ? ((((xVal > h4)) ? (xVal) : (h4))) : ((((n == 4u)) ? (xVal) : (h4))));
    float alpha = as_type<float>(0x3c23d70au);
    float oneMinusAlpha = (as_type<float>(0x3f800000u) - alpha);
    float finalH2 = (((n >= 5u)) ? (((oneMinusAlpha * h2) + (alpha * xVal))) : ((((n == 2u)) ? (xVal) : (h2))));
    q0[0] = finalH0;
    q2[0] = finalH2;
    q4[0] = finalH4;
    uint newCount = (n + 1u);
    count[0] = newCount;
  }
}
