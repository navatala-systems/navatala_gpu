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
extern "C" __global__ void navatala_sparse_knn_sparse_k_n_n_distance_f64(const double* queryData, const int* queryIndices, const unsigned int* queryStart, const unsigned int* queryEnd, const double* idxData, const int* idxIndices, const int* idxIndptr, const unsigned int* nIdxRows, double* distances) {
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
    double queryNormSqAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int qIdxAccum = qStart;
    unsigned int qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      unsigned int qPos = qIdxAccum;
      if (qPos < qEnd) {
        double qVal = queryData[qPos];
        double qValSq = (qVal * qVal);
        double currQNorm = queryNormSqAccum;
        double newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      unsigned int currQIdx = qIdxAccum;
      unsigned int nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    double queryNormSq = queryNormSqAccum;
    double idxNormSqAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int iIdxAccum = idxStart;
    unsigned int idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      unsigned int iPos = iIdxAccum;
      if (iPos < idxEnd) {
        double iVal = idxData[iPos];
        double iValSq = (iVal * iVal);
        double currINorm = idxNormSqAccum;
        double newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      unsigned int currIIdx = iIdxAccum;
      unsigned int nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    double idxNormSq = idxNormSqAccum;
    double dotAccum = __longlong_as_double(0x0000000000000000ull);
    unsigned int qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      unsigned int qPos2 = qIdx2Accum;
      if (qPos2 < qEnd) {
        double qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        unsigned int iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          unsigned int iPos2 = iIdx2Accum;
          if (iPos2 < idxEnd) {
            int iCol = idxIndices[iPos2];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              double iVal2 = idxData[iPos2];
              double prod = (qVal2 * iVal2);
              double currDot = dotAccum;
              double newDot = (currDot + prod);
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
    double dotProd = dotAccum;
    double twoDot = (__longlong_as_double(0x4000000000000000ull) * dotProd);
    double sumNorms = (queryNormSq + idxNormSq);
    double distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}
