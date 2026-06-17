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

kernel void navatala_sparse_sparse_jacobi_iter_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colInd [[buffer(1)]], device const float* values [[buffer(2)]], device const float* x [[buffer(3)]], device const float* b [[buffer(4)]], device const uint* n [[buffer(5)]], device float* x_new [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint i = gid;
  uint nVal = n[0u];
  if ((i < nVal)) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + 1u);
    uint rowEnd = rowPtr[iPlusOne];
    float sumAccum = as_type<float>(0x00000000u);
    float diagAccum = as_type<float>(0x3f800000u);
    float bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if ((jU32 >= rowStart)) {
        uint col = colInd[jU32];
        float aVal = values[jU32];
        bool isDiag = (col == i);
        float currentDiag = diagAccum;
        float newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        float xj = x[col];
        float contrib = (aVal * xj);
        float offDiagContrib = ((isDiag) ? (as_type<float>(0x00000000u)) : (contrib));
        float currentSum = sumAccum;
        float newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    float finalSum = sumAccum;
    float finalDiag = diagAccum;
    float numerator = (bi - finalSum);
    float result = (numerator / finalDiag);
    x_new[i] = result;
  }
}
