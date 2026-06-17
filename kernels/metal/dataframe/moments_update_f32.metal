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

kernel void navatala_dataframe_moments_update_f32(device const float* x [[buffer(0)]], device uint* n [[buffer(1)]], device float* M1 [[buffer(2)]], device float* M2 [[buffer(3)]], device float* M3 [[buffer(4)]], device float* M4 [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((int(__gid.x) == 0)) {
    uint nOld = n[0];
    float m1Old = M1[0];
    float m2Old = M2[0];
    float m3Old = M3[0];
    float m4Old = M4[0];
    float xVal = x[0];
    uint nNew = (nOld + 1u);
    float nNewF = ((float)(nNew));
    float nOldF = ((float)(nOld));
    float delta = (xVal - m1Old);
    float delta_n = (delta / nNewF);
    float delta_n2 = (delta_n * delta_n);
    float term1 = ((delta * delta_n) * nOldF);
    float m1New = (m1Old + delta_n);
    float nNew2 = (nNewF * nNewF);
    float threeNNew = (as_type<float>(0x40400000u) * nNewF);
    float nNew2MinusThreeN = (nNew2 - threeNNew);
    float m4Coeff = (nNew2MinusThreeN + as_type<float>(0x40400000u));
    float m4Term1 = ((term1 * delta_n2) * m4Coeff);
    float m4Term2 = ((as_type<float>(0x40c00000u) * delta_n2) * m2Old);
    float m4Term3 = ((as_type<float>(0x40800000u) * delta_n) * m3Old);
    float m4New = (((m4Old + m4Term1) + m4Term2) - m4Term3);
    float nNewMinus2 = (nNewF - as_type<float>(0x40000000u));
    float m3Term1 = ((term1 * delta_n) * nNewMinus2);
    float m3Term2 = ((as_type<float>(0x40400000u) * delta_n) * m2Old);
    float m3New = ((m3Old + m3Term1) - m3Term2);
    float m2New = (m2Old + term1);
    n[0] = nNew;
    M1[0] = m1New;
    M2[0] = m2New;
    M3[0] = m3New;
    M4[0] = m4New;
  }
}
