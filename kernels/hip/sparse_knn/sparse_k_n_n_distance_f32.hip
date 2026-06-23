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
extern "C" __global__ void navatala_sparse_knn_sparse_k_n_n_distance_f32(const float* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const float* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, float* distances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nRows = nIdxRows[0];
  unsigned int qStart = queryStart[0];
  unsigned int qEnd = queryEnd[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    unsigned int rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    unsigned int idxStart = ((unsigned int)(idxRowStart));
    unsigned int idxEnd = ((unsigned int)(idxRowEnd));
    float queryNormSqAccum = __uint_as_float(0x00000000u);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if (qPos < qEnd) {
        float qVal = queryData[qPos];
        float qValSq = (qVal * qVal);
        float currQNorm = queryNormSqAccum;
        float newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float queryNormSq = queryNormSqAccum;
    float idxNormSqAccum = __uint_as_float(0x00000000u);
    unsigned int iIdxAccum = idxStart;
    unsigned int idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      unsigned int iPos = iIdxAccum;
      if (iPos < idxEnd) {
        float iVal = idxData[iPos];
        float iValSq = (iVal * iVal);
        float currINorm = idxNormSqAccum;
        float newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      unsigned int currIIdx = iIdxAccum;
      unsigned int nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    float idxNormSq = idxNormSqAccum;
    float dotAccum = __uint_as_float(0x00000000u);
    unsigned int qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      unsigned int qPos2 = qIdx2Accum;
      if (qPos2 < qEnd) {
        float qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        unsigned int iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          unsigned int iPos2 = iIdx2Accum;
          if (iPos2 < idxEnd) {
            int iCol = idxIndices[iPos2];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal2 = idxData[iPos2];
              float prod = (qVal2 * iVal2);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          unsigned int currI2Idx = iIdx2Accum;
          unsigned int nextI2Idx = (currI2Idx + 1u);
          iIdx2Accum = nextI2Idx;
        }
      }
      unsigned int currQ2Idx = qIdx2Accum;
      unsigned int nextQ2Idx = (currQ2Idx + 1u);
      qIdx2Accum = nextQ2Idx;
    }
    float dotProd = dotAccum;
    float twoDot = (__uint_as_float(0x40000000u) * dotProd);
    float sumNorms = (queryNormSq + idxNormSq);
    float distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}
