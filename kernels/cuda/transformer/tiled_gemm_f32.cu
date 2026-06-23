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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_f32(const float* a, const float* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tidX = ((unsigned int)((int)(threadIdx.x)));
  unsigned int tidY = ((unsigned int)((int)(threadIdx.y)));
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y)));
  unsigned int mDim = m[0u];
  unsigned int nDim = n[0u];
  unsigned int kDim = k[0u];
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int tileSize = 16u;
  unsigned int row = ((gidY * tileSize) + tidY);
  unsigned int col = ((gidX * tileSize) + tidX);
  bool rowValid = (row < mDim);
  bool colValid = (col < nDim);
  bool valid = (rowValid && colValid);
  __shared__ float tileA[256];
  __shared__ float tileB[256];
  float acc = __uint_as_float(0x00000000u);
  unsigned int tileCount = ((kDim + (tileSize - 1u)) / tileSize);
  for (int tileIter = 0; tileIter < (int)(tileCount); ++tileIter) {
    unsigned int tileIterU32 = ((unsigned int)(tileIter));
    unsigned int tileBase = (tileIterU32 * tileSize);
    unsigned int aK = (tileBase + tidX);
    unsigned int bK = (tileBase + tidY);
    unsigned int tileOffset = ((tidY * tileSize) + tidX);
    bool aKValid = (aK < kDim);
    bool bKValid = (bK < kDim);
    bool aLoadValid = (rowValid && aKValid);
    bool bLoadValid = (colValid && bKValid);
    unsigned int aIdx = ((row * kDim) + aK);
    unsigned int bIdx = ((bK * nDim) + col);
    float aTileVal = ((aLoadValid) ? (a[aIdx]) : (__uint_as_float(0x00000000u)));
    float bTileVal = ((bLoadValid) ? (b[bIdx]) : (__uint_as_float(0x00000000u)));
    tileA[tileOffset] = aTileVal;
    tileB[tileOffset] = bTileVal;
    __syncthreads();
    for (int kk = 0; kk < (int)(tileSize); ++kk) {
      unsigned int kkU32 = ((unsigned int)(kk));
      unsigned int tileAIdx = ((tidY * tileSize) + kkU32);
      unsigned int tileBIdx = ((kkU32 * tileSize) + tidX);
      float aTile = tileA[tileAIdx];
      float bTile = tileB[tileBIdx];
      float prod = (aTile * bTile);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    __syncthreads();
  }
  float finalAcc = acc;
  if (valid) {
    unsigned int cIdx = ((row * nDim) + col);
    float cOld = c[cIdx];
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    c[cIdx] = result;
  }
}
