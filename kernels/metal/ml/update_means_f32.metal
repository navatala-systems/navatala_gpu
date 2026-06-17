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

kernel void navatala_ml_update_means_f32(device const float* data [[buffer(0)]], device const float* responsibilities [[buffer(1)]], device const uint* n [[buffer(2)]], device const uint* k [[buffer(3)]], device const uint* d [[buffer(4)]], device float* means [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numK = k[0];
  uint numD = d[0];
  uint kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    uint j = (gid / numD);
    uint l = (gid % numD);
    float weightedSumAccum = as_type<float>(0x00000000u);
    float respSumAccum = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(numN); ++i) {
      uint iU32 = ((uint)(i));
      uint respIdx = ((iU32 * numK) + j);
      float r_ij = responsibilities[respIdx];
      uint dataIdx = ((iU32 * numD) + l);
      float x_il = data[dataIdx];
      float weighted = (r_ij * x_il);
      float currWeightedSum = weightedSumAccum;
      float newWeightedSum = (currWeightedSum + weighted);
      weightedSumAccum = newWeightedSum;
      float currRespSum = respSumAccum;
      float newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    float weightedSum = weightedSumAccum;
    float respSum = respSumAccum;
    float meanVal = (weightedSum / respSum);
    means[gid] = meanVal;
  }
}
