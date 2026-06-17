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

kernel void navatala_sparse_knn_sparse_inner_product_f32(device const float* queryData [[buffer(0)]], device const int* queryIndices [[buffer(1)]], device const uint* queryStart [[buffer(2)]], device const uint* queryEnd [[buffer(3)]], device const float* idxData [[buffer(4)]], device const int* idxIndices [[buffer(5)]], device const int* idxIndptr [[buffer(6)]], device const uint* nIdxRows [[buffer(7)]], device float* dotProducts [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
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
    float dotAccum = as_type<float>(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    uint idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        uint iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          uint iPos = iIdxAccum;
          if ((iPos < idxEnd)) {
            int iCol = idxIndices[iPos];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal = idxData[iPos];
              float prod = (qVal * iVal);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currIIdx = iIdxAccum;
          uint nextIIdx = (currIIdx + 1u);
          iIdxAccum = nextIIdx;
        }
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}
