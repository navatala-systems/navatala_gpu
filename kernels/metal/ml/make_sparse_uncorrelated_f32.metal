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

kernel void navatala_ml_make_sparse_uncorrelated_f32(device const float* randomNormal [[buffer(0)]], device const uint* sparseMask [[buffer(1)]], device const float* scale [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device float* X [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numN = n[0];
  uint numD = d[0];
  uint totalElems = (numN * numD);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint featureIdx = (gid % numD);
    uint maskVal = sparseMask[gid];
    float randVal = randomNormal[gid];
    float scaleVal = scale[featureIdx];
    bool isNonZero = (maskVal == 1u);
    float scaledRand = (randVal * scaleVal);
    float result = ((isNonZero) ? (scaledRand) : (as_type<float>(0x00000000u)));
    X[gid] = result;
  }
}
