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

kernel void navatala_dataframe_weighted_welford_merge_f32(device const float* weightSumA [[buffer(0)]], device const float* meanA [[buffer(1)]], device const float* M2A [[buffer(2)]], device const float* weightSumB [[buffer(3)]], device const float* meanB [[buffer(4)]], device const float* M2B [[buffer(5)]], device float* weightSum [[buffer(6)]], device float* mean [[buffer(7)]], device float* M2 [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) == 0)) {
    float wA = weightSumA[0];
    float muA = meanA[0];
    float m2AVal = M2A[0];
    float wB = weightSumB[0];
    float muB = meanB[0];
    float m2BVal = M2B[0];
    float wTotal = (wA + wB);
    float delta = (muB - muA);
    float wTotalSafe = (((wTotal == as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (wTotal));
    float muTotal = (muA + (delta * (wB / wTotalSafe)));
    float deltaSq = (delta * delta);
    float wAwB = (wA * wB);
    float correction = (deltaSq * (wAwB / wTotalSafe));
    float m2Total = ((m2AVal + m2BVal) + correction);
    weightSum[0] = wTotal;
    mean[0] = muTotal;
    M2[0] = m2Total;
  }
}
