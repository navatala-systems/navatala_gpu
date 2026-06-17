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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_tiled_gemm_f16(const __half* a, const __half* b, const unsigned int* m, const unsigned int* n, const unsigned int* k, const float* alpha, const float* beta, __half* c) {
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
  if (valid) {
    float acc = __uint_as_float(0x00000000u);
    for (int kIter = 0; kIter < (int)(kDim); ++kIter) {
      unsigned int aIdx = ((row * kDim) + kIter);
      unsigned int bIdx = ((kIter * nDim) + col);
      __half aValF16 = a[aIdx];
      __half bValF16 = b[bIdx];
      float aVal = ((float)(aValF16));
      float bVal = ((float)(bValF16));
      float prod = (aVal * bVal);
      float oldAcc = acc;
      acc = (oldAcc + prod);
    }
    float finalAcc = acc;
    unsigned int cIdx = ((row * nDim) + col);
    __half cOldF16 = c[cIdx];
    float cOld = ((float)(cOldF16));
    float scaledProd = (alphaVal * finalAcc);
    float scaledOld = (betaVal * cOld);
    float result = (scaledProd + scaledOld);
    __half resultF16 = ((__half)(result));
    c[cIdx] = resultF16;
  }
}
