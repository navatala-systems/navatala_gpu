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

__kernel void navatala_transformer_tiled_gemm_backward_f32(__global const float* gradC, __global const float* b, __global const uint* m, __global const uint* n, __global const uint* k, __global float* gradA) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < kDim);
  bool valid = (rowValid && colValid);
  __local float tileGradC[256];
  __local float tileBT[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int jIter = 0; jIter < (int)(nDim); ++jIter) {
      uint gradCIdx = ((row * nDim) + jIter);
      uint bIdx = ((col * nDim) + jIter);
      float gradCVal = gradC[gradCIdx];
      float bVal = b[bIdx];
      float prod = (gradCVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint gradAIdx = ((row * kDim) + col);
    gradA[gradAIdx] = finalAcc;
  }
}
