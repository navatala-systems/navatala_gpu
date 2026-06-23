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

kernel void navatala_dataframe_iqr_f32(device const float* sortedInput [[buffer(0)]], device const uint* count [[buffer(1)]], device float* iqr [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidU32 = ((uint)(int(__gid.x)));
  if (gidU32 == 0u) {
    uint n = count[0u];
    uint nMinus1 = (n - 1u);
    float nMinus1Float = ((float)(nMinus1));
    float idxQ1 = (as_type<float>(0x3e800000u) * nMinus1Float);
    float lowerQ1Float = floor(idxQ1);
    uint lowerQ1 = ((uint)(lowerQ1Float));
    uint upperQ1Candidate = (lowerQ1 + 1u);
    uint upperQ1 = (((upperQ1Candidate < n)) ? (upperQ1Candidate) : (nMinus1));
    float fracQ1 = (idxQ1 - lowerQ1Float);
    float valLowerQ1 = sortedInput[lowerQ1];
    float valUpperQ1 = sortedInput[upperQ1];
    float oneMinusFracQ1 = (as_type<float>(0x3f800000u) - fracQ1);
    float q1 = ((oneMinusFracQ1 * valLowerQ1) + (fracQ1 * valUpperQ1));
    float idxQ3 = (as_type<float>(0x3f400000u) * nMinus1Float);
    float lowerQ3Float = floor(idxQ3);
    uint lowerQ3 = ((uint)(lowerQ3Float));
    uint upperQ3Candidate = (lowerQ3 + 1u);
    uint upperQ3 = (((upperQ3Candidate < n)) ? (upperQ3Candidate) : (nMinus1));
    float fracQ3 = (idxQ3 - lowerQ3Float);
    float valLowerQ3 = sortedInput[lowerQ3];
    float valUpperQ3 = sortedInput[upperQ3];
    float oneMinusFracQ3 = (as_type<float>(0x3f800000u) - fracQ3);
    float q3 = ((oneMinusFracQ3 * valLowerQ3) + (fracQ3 * valUpperQ3));
    float iqrVal = (q3 - q1);
    iqr[0u] = iqrVal;
  }
}
