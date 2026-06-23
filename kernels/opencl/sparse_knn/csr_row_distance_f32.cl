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

__kernel void navatala_sparse_knn_csr_row_distance_f32(__global const float* sparseData, __global const int* sparseIndices, __global const uint* sparseStart, __global const uint* sparseEnd, __global const float* denseQuery, __global const uint* nCols, __global float* distance) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  __local float sdata[256];
  float sparseNormAccum = as_float(0x00000000u);
  float dotProdAccum = as_float(0x00000000u);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if (sPos < sEnd) {
      float sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      uint sColU32 = ((uint)(sCol));
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
    uint currSIdx = sIdxAccum;
    uint nextSIdx = (currSIdx + (uint)(1u));
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = as_float(0x00000000u);
  uint dIdxAccum = (uint)(0u);
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if (dPos < nColsVal) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + (uint)(1u));
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (as_float(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    distance[(uint)(0u)] = distSq;
  }
}
