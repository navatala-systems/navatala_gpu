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

kernel void navatala_dataframe_tdigest_cdf_f32(device const float* x [[buffer(0)]], device const float* centroidMeans [[buffer(1)]], device const float* centroidWeights [[buffer(2)]], device const uint* centroidCount [[buffer(3)]], device const float* totalWeight [[buffer(4)]], device const float* minVal [[buffer(5)]], device const float* maxVal [[buffer(6)]], device float* cdf [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    float xVal = x[0];
    uint count = centroidCount[0];
    float minV = minVal[0];
    float maxV = maxVal[0];
    float range = (maxV - minV);
    float safeRange = (((range == as_type<float>(0x00000000u))) ? (as_type<float>(0x3f800000u)) : (range));
    float rawCdf = ((xVal - minV) / safeRange);
    float clampLo = (((rawCdf < as_type<float>(0x00000000u))) ? (as_type<float>(0x00000000u)) : (rawCdf));
    float clampHi = (((clampLo > as_type<float>(0x3f800000u))) ? (as_type<float>(0x3f800000u)) : (clampLo));
    bool isEmpty = (count == 0u);
    float finalCdf = ((isEmpty) ? (as_type<float>(0x00000000u)) : (clampHi));
    cdf[0] = finalCdf;
  }
}
