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

kernel void navatala_transformer_tiled_gemm_backward_f16(device const half* gradC [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* m [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device half* gradA [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tidX = ((uint)(int(__tid.x)));
  uint tidY = ((uint)(int(__tid.y)));
  uint gidX = ((uint)(int(__tgid.x)));
  uint gidY = ((uint)(int(__tgid.y)));
  uint mDim = m[0u];
  uint nDim = n[0u];
  uint kDim = k[0u];
  uint tileSize = 16u;
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  threadgroup float tileGradC[256];
  threadgroup float tileBT[256];
  if (valid) {
    float acc = as_type<float>(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      half gradCValF16 = gradC[gradCIdx];
      half bValF16 = b[bIdx];
      float gradCVal = ((float)(gradCValF16));
      float bVal = ((float)(bValF16));
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    half finalAccF16 = ((half)(finalAcc));
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAccF16;
  }
}
