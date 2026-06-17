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

kernel void navatala_ml_evaluate_fitness_f32(device const float* predictions [[buffer(0)]], device const float* targets [[buffer(1)]], device const uint* popSize [[buffer(2)]], device const uint* nSamples [[buffer(3)]], device float* fitness [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint popSizeVal = popSize[0];
  uint nSamplesVal = nSamples[0];
  bool inBounds = (gid < popSizeVal);
  if (inBounds) {
    uint baseIdx = (gid * nSamplesVal);
    float mseAccum = as_type<float>(0x00000000u);
    int nSamplesI32 = ((int)(nSamplesVal));
    for (int j = 0; j < (int)(nSamplesI32); ++j) {
      uint jU32 = ((uint)(j));
      uint predIdx = (baseIdx + jU32);
      float pred = predictions[predIdx];
      float target = targets[jU32];
      float diff = (pred - target);
      float sqErr = (diff * diff);
      float currAccum = mseAccum;
      float newAccum = (currAccum + sqErr);
      mseAccum = newAccum;
    }
    float finalSum = mseAccum;
    float nSamplesF32 = ((float)(nSamplesVal));
    float mse = (finalSum / nSamplesF32);
    fitness[gid] = mse;
  }
}
