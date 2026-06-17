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
extern "C" __global__ void navatala_ml_pairwise_cosine_f32(const float* X, const float* Y, const unsigned int* n, const unsigned int* m, const unsigned int* d, float* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int mVal = m[0];
  unsigned int dVal = d[0];
  unsigned int totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / mVal);
    unsigned int j = (gid % mVal);
    unsigned int xBase = (i * dVal);
    unsigned int yBase = (j * dVal);
    float dotProduct = __uint_as_float(0x00000000u);
    float normXSq = __uint_as_float(0x00000000u);
    float normYSq = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(dVal); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int xIdx = (xBase + kU32);
      unsigned int yIdx = (yBase + kU32);
      float xVal = X[xIdx];
      float yVal = Y[yIdx];
      float prod = (xVal * yVal);
      float currentDot = dotProduct;
      float newDot = (currentDot + prod);
      dotProduct = newDot;
      float xSq = (xVal * xVal);
      float ySq = (yVal * yVal);
      float currentNormX = normXSq;
      float newNormX = (currentNormX + xSq);
      normXSq = newNormX;
      float currentNormY = normYSq;
      float newNormY = (currentNormY + ySq);
      normYSq = newNormY;
    }
    float finalDot = dotProduct;
    float finalNormXSq = normXSq;
    float finalNormYSq = normYSq;
    float normX = sqrt(finalNormXSq);
    float normY = sqrt(finalNormYSq);
    float normProduct = (normX * normY);
    bool normIsZero = (normProduct == __uint_as_float(0x00000000u));
    if (normIsZero) {
      D[gid] = __uint_as_float(0x00000000u);
    } else {
      float cosineSim = (finalDot / normProduct);
      float cosineDist = (__uint_as_float(0x3f800000u) - cosineSim);
      D[gid] = cosineDist;
    }
  }
}
