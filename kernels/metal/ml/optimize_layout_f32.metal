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

kernel void navatala_ml_optimize_layout_f32(device const float* embedding [[buffer(0)]], device const float* gradients [[buffer(1)]], device const uint* n_samples [[buffer(2)]], device const uint* n_components [[buffer(3)]], device const float* alpha [[buffer(4)]], device float* embedding_out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  float alphaVal = alpha[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    float embVal = embedding[gid];
    float gradVal = gradients[gid];
    float gradClipped = (((gradVal > as_type<float>(0x40800000u))) ? (as_type<float>(0x40800000u)) : ((((gradVal < as_type<float>(0xc0800000u))) ? (as_type<float>(0xc0800000u)) : (gradVal))));
    float update = (alphaVal * gradClipped);
    float newVal = (embVal + update);
    embedding_out[gid] = newVal;
  }
}
