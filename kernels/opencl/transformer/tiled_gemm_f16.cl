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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_tiled_gemm_f16(__global const half* a, __global const half* b, __global const uint* m, __global const uint* n, __global const uint* k, __global const float* alpha, __global const float* beta, __global half* c) {
  int gid0 = (int)get_global_id(0);
  uint tidX = ((uint)((int)(get_local_id(0))));
  uint tidY = ((uint)((int)(get_local_id(1))));
  uint gidX = ((uint)((int)(get_group_id(0))));
  uint gidY = ((uint)((int)(get_group_id(1))));
  uint mDim = m[(uint)(0u)];
  uint nDim = n[(uint)(0u)];
  uint kDim = k[(uint)(0u)];
  float alphaVal = alpha[(uint)(0u)];
  float betaVal = beta[(uint)(0u)];
  uint tileSize = (uint)(16u);
  uint row = ((gidY * tileSize) + tidY);
  uint col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __local float tileA[256];
  __local float tileB[256];
  if (valid) {
    float acc = as_float(0x00000000u);
    for (int kIter = 0; kIter < (int)(kDim); ++kIter) {
      uint aIdx = ((row * kDim) + kIter);
      uint bIdx = ((kIter * nDim) + col);
      half aValF16 = a[aIdx];
      half bValF16 = b[bIdx];
      float aVal = ((float)(aValF16));
      float bVal = ((float)(bValF16));
      float prod = (aVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    uint cIdx = ((row * nDim) + col);
    half cOldF16 = c[cIdx];
    float cOld = ((float)(cOldF16));
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    half resultF16 = ((half)(result));
    c[cIdx] = resultF16;
  }
}
