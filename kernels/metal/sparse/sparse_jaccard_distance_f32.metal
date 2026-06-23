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

kernel void navatala_sparse_sparse_jaccard_distance_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* nnz [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup uint sdataIntersect[256];
  threadgroup uint sdataUnion[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lidU32 = ((uint)(int(__tid.x)));
  uint nnzVal = nnz[0u];
  uint localIntersectAccum = 0u;
  uint localUnionAccum = 0u;
  uint idxAccum = gid;
  uint gridSize = ((uint)(int(__grid_size.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      bool aNonZero = (aVal != as_type<float>(0x00000000u));
      bool bNonZero = (bVal != as_type<float>(0x00000000u));
      bool inIntersect = (aNonZero && bNonZero);
      bool inUnion = (aNonZero || bNonZero);
      uint intersectInc = ((inIntersect) ? (1u) : (0u));
      uint unionInc = ((inUnion) ? (1u) : (0u));
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint strideAccum = 128u;
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
    uint nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lidU32 == 0u) {
    uint totalIntersect = sdataIntersect[0u];
    uint totalUnion = sdataUnion[0u];
    bool unionIsZero = (totalUnion == 0u);
    float intersectF = ((float)(totalIntersect));
    float unionF = ((float)(totalUnion));
    float jaccardSim = ((unionIsZero) ? (as_type<float>(0x00000000u)) : ((intersectF / unionF)));
    float jaccardDist = (as_type<float>(0x3f800000u) - jaccardSim);
    float finalResult = ((unionIsZero) ? (as_type<float>(0x00000000u)) : (jaccardDist));
    result[0u] = finalResult;
  }
}
