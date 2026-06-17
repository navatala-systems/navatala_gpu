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

kernel void navatala_ml_make_blobs_anisotropic_f32(device const float* centers [[buffer(0)]], device const float* scales [[buffer(1)]], device const uint* labels [[buffer(2)]], device const float* randomNormal [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* d [[buffer(5)]], device float* data [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint sampleIdx = (gid / numD);
    uint dimIdx = (gid % numD);
    uint label = labels[sampleIdx];
    uint paramIdx = ((label * numD) + dimIdx);
    float centerVal = centers[paramIdx];
    float scaleVal = scales[paramIdx];
    float randVal = randomNormal[gid];
    float noise = (scaleVal * randVal);
    float result = (centerVal + noise);
    data[gid] = result;
  }
}
