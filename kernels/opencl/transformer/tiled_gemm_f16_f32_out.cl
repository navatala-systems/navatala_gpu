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
__kernel void navatala_transformer_tiled_gemm_f16_f32_out(__global const half* a, __global const half* b, __global const uint* m, __global const uint* n, __global const uint* k, __global const float* alpha, __global const float* beta, __global float* c) {
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
  float acc = as_float(0x00000000u);
  uint tileCount = ((kDim + (tileSize - (uint)(1u))) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    uint tileIterU32 = ((uint)(tileIter));
    uint tileBase = (tileIterU32 * tileSize);
    uint aK = (tileBase + tidX);
    uint bK = (tileBase + tidY);
    uint tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    uint aIdx = ((row * kDim) + aK);
    uint bIdx = ((bK * nDim) + col);
    half aValF16 = ((aLoadValid) ? (a[aIdx]) : ((half)(0.000000f)));
    half bValF16 = ((bLoadValid) ? (b[bIdx]) : ((half)(0.000000f)));
    float aTileVal = ((float)(aValF16));
    float bTileVal = ((float)(bValF16));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      uint kkU32 = ((uint)(kk));
      uint tileAIdx = ((tidY * tileSize) + kkU32);
      uint tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float finalAcc = acc;
  if (valid) {
    uint cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}
