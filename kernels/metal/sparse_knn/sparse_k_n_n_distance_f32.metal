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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_knn_sparse_k_n_n_distance_f32(device const float* queryData [[buffer(0)]], device const int* queryIndices [[buffer(1)]], device const uint* queryStart [[buffer(2)]], device const uint* queryEnd [[buffer(3)]], device const float* idxData [[buffer(4)]], device const int* idxIndices [[buffer(5)]], device const int* idxIndptr [[buffer(6)]], device const uint* nIdxRows [[buffer(7)]], device float* distances [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    float queryNormSqAccum = as_type<float>(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        float qValSq = (qVal * qVal);
        float currQNorm = queryNormSqAccum;
        float newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float queryNormSq = queryNormSqAccum;
    float idxNormSqAccum = as_type<float>(0x00000000u);
    uint iIdxAccum = idxStart;
    uint idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      uint iPos = iIdxAccum;
      if ((iPos < idxEnd)) {
        float iVal = idxData[iPos];
        float iValSq = (iVal * iVal);
        float currINorm = idxNormSqAccum;
        float newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      uint currIIdx = iIdxAccum;
      uint nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    float idxNormSq = idxNormSqAccum;
    float dotAccum = as_type<float>(0x00000000u);
    uint qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      uint qPos2 = qIdx2Accum;
      if ((qPos2 < qEnd)) {
        float qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        uint iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          uint iPos2 = iIdx2Accum;
          if ((iPos2 < idxEnd)) {
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
          uint currI2Idx = iIdx2Accum;
          uint nextI2Idx = (currI2Idx + 1u);
          iIdx2Accum = nextI2Idx;
        }
      }
      uint currQ2Idx = qIdx2Accum;
      uint nextQ2Idx = (currQ2Idx + 1u);
      qIdx2Accum = nextQ2Idx;
    }
    float dotProd = dotAccum;
    float twoDot = (as_type<float>(0x40000000u) * dotProd);
    float sumNorms = (queryNormSq + idxNormSq);
    float distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}
