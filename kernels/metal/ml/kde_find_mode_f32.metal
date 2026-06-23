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

kernel void navatala_ml_kde_find_mode_f32(device const float* densities [[buffer(0)]], device const float* queryPoints [[buffer(1)]], device const uint* m [[buffer(2)]], device float* modeValue [[buffer(3)]], device float* modeLocation [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  if (gid == 0u) {
    uint numPoints = m[0];
    float maxDensity = densities[0];
    float maxLocation = queryPoints[0];
    for (int i = 0; i < (int)(numPoints); ++i) {
      uint iU32 = ((uint)(i));
      float d = densities[iU32];
      float loc = queryPoints[iU32];
      float currMaxDensity = maxDensity;
      float currMaxLocation = maxLocation;
      bool isGreater = (d > currMaxDensity);
      float newMaxDensity = ((isGreater) ? (d) : (currMaxDensity));
      float newMaxLocation = ((isGreater) ? (loc) : (currMaxLocation));
      maxDensity = newMaxDensity;
      maxLocation = newMaxLocation;
    }
    float finalMaxDensity = maxDensity;
    float finalMaxLocation = maxLocation;
    modeValue[0] = finalMaxDensity;
    modeLocation[0] = finalMaxLocation;
  }
}
