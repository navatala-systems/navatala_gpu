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
extern "C" __global__ void navatala_sparse_knn_csr_row_distance_f64(const double* sparseData, const int* sparseIndices, const unsigned int* sparseStart, const unsigned int* sparseEnd, const double* denseQuery, const unsigned int* nCols, double* distance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int sStart = sparseStart[0];
  unsigned int sEnd = sparseEnd[0];
  unsigned int nColsVal = nCols[0];
  __shared__ double sdata[256];
  double sparseNormAccum = __longlong_as_double(0x0000000000000000ull);
  double dotProdAccum = __longlong_as_double(0x0000000000000000ull);
  unsigned int sIdxAccum = sStart;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  unsigned int sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    unsigned int sPos = sIdxAccum;
    if (sPos < sEnd) {
      double sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      unsigned int sColU32 = ((unsigned int)(sCol));
      double sValSq = (sVal * sVal);
      double currSNorm = sparseNormAccum;
      double newSNorm = (currSNorm + sValSq);
      sparseNormAccum = newSNorm;
      double dVal = denseQuery[sColU32];
      double prod = (sVal * dVal);
      double currDot = dotProdAccum;
      double newDot = (currDot + prod);
      dotProdAccum = newDot;
    }
    unsigned int currSIdx = sIdxAccum;
    unsigned int nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  double denseNormAccum = __longlong_as_double(0x0000000000000000ull);
  unsigned int dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    unsigned int dPos = dIdxAccum;
    if (dPos < nColsVal) {
      double dValLoop = denseQuery[dPos];
      double dValSq = (dValLoop * dValLoop);
      double currDNorm = denseNormAccum;
      double newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    unsigned int currDIdx = dIdxAccum;
    unsigned int nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  double sparseNormSq = sparseNormAccum;
  double denseNormSq = denseNormAccum;
  double dotProd = dotProdAccum;
  double twoDot = (__longlong_as_double(0x4000000000000000ull) * dotProd);
  double sumNorms = (sparseNormSq + denseNormSq);
  double distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}
