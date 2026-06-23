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

kernel void navatala_dataframe_median_f32(device const float* sortedInput [[buffer(0)]], device const uint* count [[buffer(1)]], device float* median [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gidU32 = ((uint)(int(__gid.x)));
  if (gidU32 == 0u) {
    uint n = count[0u];
    uint nMinus1 = (n - 1u);
    float nMinus1Float = ((float)(nMinus1));
    float idx = (as_type<float>(0x3f000000u) * nMinus1Float);
    float lowerFloat = floor(idx);
    uint lower = ((uint)(lowerFloat));
    uint upperCandidate = (lower + 1u);
    uint upper = (((upperCandidate < n)) ? (upperCandidate) : (nMinus1));
    float frac = (idx - lowerFloat);
    float valLower = sortedInput[lower];
    float valUpper = sortedInput[upper];
    float oneMinusFrac = (as_type<float>(0x3f800000u) - frac);
    float termLower = (oneMinusFrac * valLower);
    float termUpper = (frac * valUpper);
    float medianVal = (termLower + termUpper);
    median[0u] = medianVal;
  }
}
