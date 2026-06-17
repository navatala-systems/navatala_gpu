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

kernel void navatala_ml_project_to_principal_f32(device const float* X [[buffer(0)]], device const float* V [[buffer(1)]], device const uint* nSamples [[buffer(2)]], device const uint* nFeatures [[buffer(3)]], device const uint* nComponents [[buffer(4)]], device float* Z [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  uint totalOutput = (n * kComp);
  bool inBounds = (gid < totalOutput);
  if (inBounds) {
    uint i = (gid / kComp);
    uint c = (gid % kComp);
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(d); ++j) {
      uint jU32 = ((uint)(j));
      uint xIdx = ((i * d) + jU32);
      uint vIdx = ((c * d) + jU32);
      float xVal = X[xIdx];
      float vVal = V[vIdx];
      float prod = (xVal * vVal);
      float currSum = sum;
      float newSum = (currSum + prod);
      sum = newSum;
    }
    float finalSum = sum;
    Z[gid] = finalSum;
  }
}
