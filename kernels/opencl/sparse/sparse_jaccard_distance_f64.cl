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
__kernel void navatala_sparse_sparse_jaccard_distance_f64(__global const double* a, __global const double* b, __global const uint* nnz, __global double* result) {
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
      double aVal = a[currentIdx];
      double bVal = b[currentIdx];
      bool aNonZero = (aVal != as_double(0x0000000000000000ul));
      bool bNonZero = (bVal != as_double(0x0000000000000000ul));
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
    double intersectF = ((double)(totalIntersect));
    double unionF = ((double)(totalUnion));
    double jaccardSim = ((unionIsZero) ? (as_double(0x0000000000000000ul)) : ((intersectF / unionF)));
    double jaccardDist = (as_double(0x3ff0000000000000ul) - jaccardSim);
    double finalResult = ((unionIsZero) ? (as_double(0x0000000000000000ul)) : (jaccardDist));
    result[(uint)(0u)] = finalResult;
  }
}
