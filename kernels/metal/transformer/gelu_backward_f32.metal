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

kernel void navatala_transformer_gelu_backward_f32(device const float* _input [[buffer(0)]], device const float* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device float* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    float dy = gradOutput[gid];
    float x2 = (x * x);
    float x3 = (x2 * x);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (x + cubeTerm);
    float u = (as_type<float>(0x3f4c422au) * inner);
    float tanhU = tanh(u);
    float tanh2 = (tanhU * tanhU);
    float sech2 = (as_type<float>(0x3f800000u) - tanh2);
    float threeCoef = as_type<float>(0x3e095d4fu);
    float derivInner = (as_type<float>(0x3f800000u) + (threeCoef * x2));
    float term2a = (as_type<float>(0x3f000000u) * x);
    float term2b = (term2a * sech2);
    float term2c = (term2b * as_type<float>(0x3f4c422au));
    float term2 = (term2c * derivInner);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhU);
    float term1 = (as_type<float>(0x3f000000u) * onePlusTanh);
    float geluPrime = (term1 + term2);
    float dx = (dy * geluPrime);
    gradInput[gid] = dx;
  }
}
