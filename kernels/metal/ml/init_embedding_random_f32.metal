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

kernel void navatala_ml_init_embedding_random_f32(device const uint* nPoints [[buffer(0)]], device const uint* outputDims [[buffer(1)]], device const uint* seed [[buffer(2)]], device float* Y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint s = seed[0];
  uint totalElems = (n * d);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint state = (s + gid);
    uint lcgA = 48271u;
    uint state1 = (lcgA * state);
    uint state2 = (lcgA * state1);
    float maxVal = as_type<float>(0x4f000000u);
    float u1 = (((float)(state1)) / maxVal);
    float u2 = (((float)(state2)) / maxVal);
    float negTwoLnU1 = (as_type<float>(0xc0000000u) * log(u1));
    float sqrtTerm = sqrt(negTwoLnU1);
    float twoPiU2 = (as_type<float>(0x40c90fdbu) * u2);
    float z = (sqrtTerm * cos(twoPiU2));
    float yVal = (z * as_type<float>(0x3c23d70au));
    Y[gid] = yVal;
  }
}
