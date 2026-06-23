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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_knn_sparse_k_n_n_distance_f64(__global const double* queryData, __global const int* queryIndices, __global const uint* queryStart, __global const uint* queryEnd, __global const double* idxData, __global const int* idxIndices, __global const int* idxIndptr, __global const uint* nIdxRows, __global double* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + (uint)(1u));
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    double queryNormSqAccum = as_double(0x0000000000000000ul);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if (qPos < qEnd) {
        double qVal = queryData[qPos];
        double qValSq = (qVal * qVal);
        double currQNorm = queryNormSqAccum;
        double newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + (uint)(1u));
      qIdxAccum = nextQIdx;
    }
    double queryNormSq = queryNormSqAccum;
    double idxNormSqAccum = as_double(0x0000000000000000ul);
    uint iIdxAccum = idxStart;
    uint idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      uint iPos = iIdxAccum;
      if (iPos < idxEnd) {
        double iVal = idxData[iPos];
        double iValSq = (iVal * iVal);
        double currINorm = idxNormSqAccum;
        double newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      uint currIIdx = iIdxAccum;
      uint nextIIdx = (currIIdx + (uint)(1u));
      iIdxAccum = nextIIdx;
    }
    double idxNormSq = idxNormSqAccum;
    double dotAccum = as_double(0x0000000000000000ul);
    uint qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      uint qPos2 = qIdx2Accum;
      if (qPos2 < qEnd) {
        double qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        uint iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          uint iPos2 = iIdx2Accum;
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
          uint currI2Idx = iIdx2Accum;
          uint nextI2Idx = (currI2Idx + (uint)(1u));
          iIdx2Accum = nextI2Idx;
        }
      }
      uint currQ2Idx = qIdx2Accum;
      uint nextQ2Idx = (currQ2Idx + (uint)(1u));
      qIdx2Accum = nextQ2Idx;
    }
    double dotProd = dotAccum;
    double twoDot = (as_double(0x4000000000000000ul) * dotProd);
    double sumNorms = (queryNormSq + idxNormSq);
    double distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}
