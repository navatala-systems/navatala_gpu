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
extern "C" __global__ void navatala_sparse_knn_csr_row_distance_f32(const float* sparseData, const int* sparseIndices, const unsigned int* sparseStart, const unsigned int* sparseEnd, const float* denseQuery, const unsigned int* nCols, float* distance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int sStart = sparseStart[0];
  unsigned int sEnd = sparseEnd[0];
  unsigned int nColsVal = nCols[0];
  __shared__ float sdata[256];
  float sparseNormAccum = __uint_as_float(0x00000000u);
  float dotProdAccum = __uint_as_float(0x00000000u);
  unsigned int sIdxAccum = sStart;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  unsigned int sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    unsigned int sPos = sIdxAccum;
    if (sPos < sEnd) {
      float sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      unsigned int sColU32 = ((unsigned int)(sCol));
      float sValSq = (sVal * sVal);
      float currSNorm = sparseNormAccum;
      float newSNorm = (currSNorm + sValSq);
      sparseNormAccum = newSNorm;
      float dVal = denseQuery[sColU32];
      float prod = (sVal * dVal);
      float currDot = dotProdAccum;
      float newDot = (currDot + prod);
      dotProdAccum = newDot;
    }
    unsigned int currSIdx = sIdxAccum;
    unsigned int nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = __uint_as_float(0x00000000u);
  unsigned int dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    unsigned int dPos = dIdxAccum;
    if (dPos < nColsVal) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    unsigned int currDIdx = dIdxAccum;
    unsigned int nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (__uint_as_float(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}
