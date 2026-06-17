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

kernel void navatala_ml_kde_evaluate_gaussian_f32(device const float* data [[buffer(0)]], device const float* queryPoints [[buffer(1)]], device const float* bandwidth [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* m [[buffer(4)]], device float* densities [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    uint numSamples = n[0];
    float h = bandwidth[0];
    float x = queryPoints[gid];
    float invSqrt2Pi = as_type<float>(0x3ecc422au);
    float sum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numSamples); ++i) {
      uint iU32 = ((uint)(i));
      float xi = data[iU32];
      float diff = (x - xi);
      float u = (diff / h);
      float uSq = (u * u);
      float negHalfUSq = (as_type<float>(0xbf000000u) * uSq);
      float expTerm = exp(negHalfUSq);
      float kernelVal = (invSqrt2Pi * expTerm);
      float currSum = sum;
      sum = (currSum + kernelVal);
    }
    float nFloat = ((float)(numSamples));
    float nh = (nFloat * h);
    float finalSum = sum;
    float density = (finalSum / nh);
    densities[gid] = density;
  }
}
