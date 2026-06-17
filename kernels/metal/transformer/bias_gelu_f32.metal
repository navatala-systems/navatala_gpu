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

kernel void navatala_transformer_bias_gelu_f32(device const float* _input [[buffer(0)]], device const float* bias [[buffer(1)]], device const uint* count [[buffer(2)]], device const uint* biasSize [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint bSize = biasSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint biasIdx = (gid % bSize);
    float b = bias[biasIdx];
    float xBias = (x + b);
    float x2 = (xBias * xBias);
    float x3 = (x2 * xBias);
    float cubeTerm = (as_type<float>(0x3d372713u) * x3);
    float inner = (xBias + cubeTerm);
    float tanhArg = (as_type<float>(0x3f4c422au) * inner);
    float tanhVal = tanh(tanhArg);
    float onePlusTanh = (as_type<float>(0x3f800000u) + tanhVal);
    float halfXBias = (as_type<float>(0x3f000000u) * xBias);
    float result = (halfXBias * onePlusTanh);
    _output[gid] = result;
  }
}
