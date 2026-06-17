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

kernel void navatala_transformer_quick_gelu_backward_f16(device const half* _input [[buffer(0)]], device const half* gradOutput [[buffer(1)]], device const uint* count [[buffer(2)]], device half* gradInput [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    half x = _input[gid];
    half dy = gradOutput[gid];
    float xF32 = ((float)(x));
    float dyF32 = ((float)(dy));
    float scaledX = (as_type<float>(0x3fd9db23u) * xF32);
    float negScaledX = (-scaledX);
    float expNegScaledX = exp(negScaledX);
    float denom = (as_type<float>(0x3f800000u) + expNegScaledX);
    float sigmoid = (as_type<float>(0x3f800000u) / denom);
    float oneMinusSigmoid = (as_type<float>(0x3f800000u) - sigmoid);
    float term2a = (scaledX * sigmoid);
    float term2 = (term2a * oneMinusSigmoid);
    float quickGeluPrime = (sigmoid + term2);
    float dxF32 = (dyF32 * quickGeluPrime);
    half dx = ((half)(dxF32));
    gradInput[gid] = dx;
  }
}
