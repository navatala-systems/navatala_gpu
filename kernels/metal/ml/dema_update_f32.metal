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

kernel void navatala_ml_dema_update_f32(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device float* ema1 [[buffer(2)]], device float* ema2 [[buffer(3)]], device uint* count [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) == 0) {
    float xVal = x[0];
    float a = alpha[0];
    float old1 = ema1[0];
    float old2 = ema2[0];
    uint n = count[0];
    float omA = (as_type<float>(0x3f800000u) - a);
    float aX = (a * xVal);
    float omA1 = (omA * old1);
    float e1u = (aX + omA1);
    float new1 = (((n == 0u)) ? (xVal) : (e1u));
    float aE1 = (a * new1);
    float omA2 = (omA * old2);
    float e2u = (aE1 + omA2);
    float new2 = (((n == 0u)) ? (new1) : (e2u));
    ema1[0] = new1;
    ema2[0] = new2;
    uint nc = (n + 1u);
    count[0] = nc;
  }
}
