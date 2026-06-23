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
__kernel void navatala_sparse_knn_csr_row_distance_f64(__global const double* sparseData, __global const int* sparseIndices, __global const uint* sparseStart, __global const uint* sparseEnd, __global const double* denseQuery, __global const uint* nCols, __global double* distance) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  __local double sdata[256];
  double sparseNormAccum = as_double(0x0000000000000000ul);
  double dotProdAccum = as_double(0x0000000000000000ul);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if (sPos < sEnd) {
      double sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      uint sColU32 = ((uint)(sCol));
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
    uint currSIdx = sIdxAccum;
    uint nextSIdx = (currSIdx + (uint)(1u));
    sIdxAccum = nextSIdx;
  }
  double denseNormAccum = as_double(0x0000000000000000ul);
  uint dIdxAccum = (uint)(0u);
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if (dPos < nColsVal) {
      double dValLoop = denseQuery[dPos];
      double dValSq = (dValLoop * dValLoop);
      double currDNorm = denseNormAccum;
      double newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + (uint)(1u));
    dIdxAccum = nextDIdx;
  }
  double sparseNormSq = sparseNormAccum;
  double denseNormSq = denseNormAccum;
  double dotProd = dotProdAccum;
  double twoDot = (as_double(0x4000000000000000ul) * dotProd);
  double sumNorms = (sparseNormSq + denseNormSq);
  double distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == (uint)(0u));
  if (isThread0) {
    distance[(uint)(0u)] = distSq;
  }
}
