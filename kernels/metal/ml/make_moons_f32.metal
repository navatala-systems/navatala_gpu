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

kernel void navatala_ml_make_moons_f32(device const uint* labels [[buffer(0)]], device const float* randomUniform [[buffer(1)]], device const float* randomNormal [[buffer(2)]], device const float* noise [[buffer(3)]], device const uint* n [[buffer(4)]], device float* data [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float sigma = noise[0];
    uint label = labels[gid];
    float u = randomUniform[gid];
    float theta = (as_type<float>(0x40490fdbu) * u);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    uint outIdxX = (gid * 2u);
    uint outIdxY = (outIdxX + 1u);
    float noiseX = randomNormal[outIdxX];
    float noiseY = randomNormal[outIdxY];
    bool isLowerMoon = (label == 1u);
    float baseX_upper = cosTheta;
    float baseY_upper = sinTheta;
    float baseX_lower = (as_type<float>(0x3f800000u) - cosTheta);
    float baseY_lower = (as_type<float>(0x3f000000u) - sinTheta);
    float baseX = ((isLowerMoon) ? (baseX_lower) : (baseX_upper));
    float baseY = ((isLowerMoon) ? (baseY_lower) : (baseY_upper));
    float resultX = (baseX + (sigma * noiseX));
    float resultY = (baseY + (sigma * noiseY));
    data[outIdxX] = resultX;
    data[outIdxY] = resultY;
  }
}
