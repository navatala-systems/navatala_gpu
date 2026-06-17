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

kernel void navatala_sparse_knn_csr_row_distance_f32(device const float* sparseData [[buffer(0)]], device const int* sparseIndices [[buffer(1)]], device const uint* sparseStart [[buffer(2)]], device const uint* sparseEnd [[buffer(3)]], device const float* denseQuery [[buffer(4)]], device const uint* nCols [[buffer(5)]], device float* distance [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  threadgroup float sdata[256];
  float sparseNormAccum = as_type<float>(0x00000000u);
  float dotProdAccum = as_type<float>(0x00000000u);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)(int(__grid_size.x)));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if ((sPos < sEnd)) {
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
    uint nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = as_type<float>(0x00000000u);
  uint dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if ((dPos < nColsVal)) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (as_type<float>(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}
