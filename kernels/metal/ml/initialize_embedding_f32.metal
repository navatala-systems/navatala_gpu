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

kernel void navatala_ml_initialize_embedding_f32(device const uint* seed [[buffer(0)]], device const uint* n_samples [[buffer(1)]], device const uint* n_components [[buffer(2)]], device const float* spread [[buffer(3)]], device float* embedding [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  uint seedVal = seed[0];
  float spreadVal = spread[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    uint state = (seedVal + gid);
    uint a = 1664525u;
    uint c = 1013904223u;
    uint newState = ((a * state) + c);
    float randBitsF = ((float)(newState));
    float maxU32 = as_type<float>(0x4f800000u);
    float normalized = (randBitsF / maxU32);
    float centered = ((normalized * as_type<float>(0x40000000u)) - as_type<float>(0x3f800000u));
    float scaled = (centered * spreadVal);
    embedding[gid] = scaled;
  }
}
