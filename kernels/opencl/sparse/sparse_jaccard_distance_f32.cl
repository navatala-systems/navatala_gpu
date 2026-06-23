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

__kernel void navatala_sparse_sparse_jaccard_distance_f32(__global const float* a, __global const float* b, __global const uint* nnz, __global float* result) {
  int gid0 = (int)get_global_id(0);
  __local uint sdataIntersect[256];
  __local uint sdataUnion[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lidU32 = ((uint)((int)(get_local_id(0))));
  uint nnzVal = nnz[(uint)(0u)];
  uint localIntersectAccum = (uint)(0u);
  uint localUnionAccum = (uint)(0u);
  uint idxAccum = gid;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      bool aNonZero = (aVal != as_float(0x00000000u));
      bool bNonZero = (bVal != as_float(0x00000000u));
      bool inIntersect = (aNonZero && bNonZero);
      bool inUnion = (aNonZero || bNonZero);
      uint intersectInc = ((inIntersect) ? ((uint)(1u)) : ((uint)(0u)));
      uint unionInc = ((inUnion) ? ((uint)(1u)) : ((uint)(0u)));
      uint currentIntersect = localIntersectAccum;
      uint newIntersect = (currentIntersect + intersectInc);
      localIntersectAccum = newIntersect;
      uint currentUnion = localUnionAccum;
      uint newUnion = (currentUnion + unionInc);
      localUnionAccum = newUnion;
    }
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  uint finalLocalIntersect = localIntersectAccum;
  uint finalLocalUnion = localUnionAccum;
  sdataIntersect[lidU32] = finalLocalIntersect;
  sdataUnion[lidU32] = finalLocalUnion;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint strideAccum = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if (lidU32 < currentStride) {
      uint otherIdx = (lidU32 + currentStride);
      uint myIntersect = sdataIntersect[lidU32];
      uint otherIntersect = sdataIntersect[otherIdx];
      sdataIntersect[lidU32] = (myIntersect + otherIntersect);
      uint myUnion = sdataUnion[lidU32];
      uint otherUnion = sdataUnion[otherIdx];
      sdataUnion[lidU32] = (myUnion + otherUnion);
    }
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> (uint)(1u));
    strideAccum = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lidU32 == (uint)(0u)) {
    uint totalIntersect = sdataIntersect[(uint)(0u)];
    uint totalUnion = sdataUnion[(uint)(0u)];
    bool unionIsZero = (totalUnion == (uint)(0u));
    float intersectF = ((float)(totalIntersect));
    float unionF = ((float)(totalUnion));
    float jaccardSim = ((unionIsZero) ? (as_float(0x00000000u)) : ((intersectF / unionF)));
    float jaccardDist = (as_float(0x3f800000u) - jaccardSim);
    float finalResult = ((unionIsZero) ? (as_float(0x00000000u)) : (jaccardDist));
    result[(uint)(0u)] = finalResult;
  }
}
