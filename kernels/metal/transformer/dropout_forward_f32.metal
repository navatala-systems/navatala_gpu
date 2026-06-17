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

kernel void navatala_transformer_dropout_forward_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const float* prob [[buffer(2)]], device const uint* seed [[buffer(3)]], device float* _output [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  float p = prob[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float x = _input[gid];
    uint h0 = (gid ^ s);
    uint h1 = (h0 * 3432918353u);
    uint h2 = (h1 ^ (h1 >> 15u));
    uint h3 = (h2 * 461845907u);
    uint hashVal = (h3 ^ (h3 >> 13u));
    float randF32 = (((float)(hashVal)) / as_type<float>(0x4f800000u));
    bool keep = (randF32 > p);
    float oneMinusP = (as_type<float>(0x3f800000u) - p);
    float scale = (as_type<float>(0x3f800000u) / oneMinusP);
    float scaled = (x * scale);
    float result = ((keep) ? (scaled) : (as_type<float>(0x00000000u)));
    _output[gid] = result;
  }
}
