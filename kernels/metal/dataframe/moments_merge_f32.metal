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

kernel void navatala_dataframe_moments_merge_f32(device const uint* nA [[buffer(0)]], device const float* M1A [[buffer(1)]], device const float* M2A [[buffer(2)]], device const float* M3A [[buffer(3)]], device const float* M4A [[buffer(4)]], device const uint* nB [[buffer(5)]], device const float* M1B [[buffer(6)]], device const float* M2B [[buffer(7)]], device const float* M3B [[buffer(8)]], device const float* M4B [[buffer(9)]], device uint* nOut [[buffer(10)]], device float* M1Out [[buffer(11)]], device float* M2Out [[buffer(12)]], device float* M3Out [[buffer(13)]], device float* M4Out [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    uint nAVal = nA[0];
    float m1AVal = M1A[0];
    float m2AVal = M2A[0];
    float m3AVal = M3A[0];
    float m4AVal = M4A[0];
    uint nBVal = nB[0];
    float m1BVal = M1B[0];
    float m2BVal = M2B[0];
    float m3BVal = M3B[0];
    float m4BVal = M4B[0];
    uint nTotal = (nAVal + nBVal);
    float nAf = ((float)(nAVal));
    float nBf = ((float)(nBVal));
    float nTotalf = ((float)(nTotal));
    float nTotalSafe = (((nTotal == 0u)) ? (as_type<float>(0x3f800000u)) : (nTotalf));
    float delta = (m1BVal - m1AVal);
    float delta2 = (delta * delta);
    float delta3 = (delta2 * delta);
    float delta4 = (delta3 * delta);
    float nAnB = (nAf * nBf);
    float m1New = (m1AVal + (delta * (nBf / nTotalSafe)));
    float m2Term = ((delta2 * nAnB) / nTotalSafe);
    float m2New = ((m2AVal + m2BVal) + m2Term);
    float n2 = (nTotalSafe * nTotalSafe);
    float nAMinusNB = (nAf - nBf);
    float m3Term1 = (((delta3 * nAnB) * nAMinusNB) / n2);
    float nAM2B = (nAf * m2BVal);
    float nBM2A = (nBf * m2AVal);
    float m3Term2 = (((as_type<float>(0x40400000u) * delta) * (nAM2B - nBM2A)) / nTotalSafe);
    float m3New = (((m3AVal + m3BVal) + m3Term1) + m3Term2);
    float n3 = (n2 * nTotalSafe);
    float nA2 = (nAf * nAf);
    float nB2 = (nBf * nBf);
    float nA2MinusNAnB = (nA2 - nAnB);
    float m4Coeff = (nA2MinusNAnB + nB2);
    float m4Term1 = (((delta4 * nAnB) * m4Coeff) / n3);
    float nA2M2B = (nA2 * m2BVal);
    float nB2M2A = (nB2 * m2AVal);
    float m4Term2 = (((as_type<float>(0x40c00000u) * delta2) * (nA2M2B + nB2M2A)) / n2);
    float nAM3B = (nAf * m3BVal);
    float nBM3A = (nBf * m3AVal);
    float m4Term3 = (((as_type<float>(0x40800000u) * delta) * (nAM3B - nBM3A)) / nTotalSafe);
    float m4New = ((((m4AVal + m4BVal) + m4Term1) + m4Term2) + m4Term3);
    nOut[0] = nTotal;
    M1Out[0] = m1New;
    M2Out[0] = m2New;
    M3Out[0] = m3New;
    M4Out[0] = m4New;
  }
}
