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

kernel void navatala_sparse_chebyshev_smooth_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* xPrev [[buffer(4)]], device const float* diagInv [[buffer(5)]], device const float* b [[buffer(6)]], device const uint* n [[buffer(7)]], device const float* alpha [[buffer(8)]], device const float* beta [[buffer(9)]], device float* xNext [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float term = (aVal * xj);
        float currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    float bi = b[i];
    float xi = x[i];
    float xPrevI = xPrev[i];
    float dInv = diagInv[i];
    float alphaVal = alpha[0u];
    float betaVal = beta[0u];
    float residual = (bi - sumAccum);
    float scaledResidual = (dInv * residual);
    float history = (xi - xPrevI);
    float update = ((alphaVal * scaledResidual) + (betaVal * history));
    xNext[i] = (xi + update);
  }
}
