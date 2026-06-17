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

kernel void navatala_dataframe_adjusted_rand_index_f32(device const float* contingency [[buffer(0)]], device const float* rowSums [[buffer(1)]], device const float* colSums [[buffer(2)]], device const float* n [[buffer(3)]], device const uint* numTrue [[buffer(4)]], device const uint* numPred [[buffer(5)]], device float* ari [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sumNij[256];
  threadgroup float sumAi[256];
  threadgroup float sumBj[256];
  float nVal = n[0u];
  uint numTrueVal = numTrue[0u];
  uint numPredVal = numPred[0u];
  uint totalCells = (numTrueVal * numPredVal);
  float localSumNijAccum = as_type<float>(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      float nij = contingency[idxU32];
      float nijM1 = (nij - as_type<float>(0x3f800000u));
      float choose2 = ((nij * nijM1) / as_type<float>(0x40000000u));
      float curSumNij = localSumNijAccum;
      float newSumNij = (curSumNij + choose2);
      localSumNijAccum = newSumNij;
    }
  }
  float localSumNij = localSumNijAccum;
  float localSumAiAccum = as_type<float>(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      float aiM1 = (ai - as_type<float>(0x3f800000u));
      float choose2Ai = ((ai * aiM1) / as_type<float>(0x40000000u));
      float curSumAi = localSumAiAccum;
      float newSumAi = (curSumAi + choose2Ai);
      localSumAiAccum = newSumAi;
    }
  }
  float localSumAi = localSumAiAccum;
  float localSumBjAccum = as_type<float>(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      float bjM1 = (bj - as_type<float>(0x3f800000u));
      float choose2Bj = ((bj * bjM1) / as_type<float>(0x40000000u));
      float curSumBj = localSumBjAccum;
      float newSumBj = (curSumBj + choose2Bj);
      localSumBjAccum = newSumBj;
    }
  }
  float localSumBj = localSumBjAccum;
  sumNij[lid] = localSumNij;
  sumAi[lid] = localSumAi;
  sumBj[lid] = localSumBj;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int stride = 0; stride < (int)(128u); ++stride) {
    uint strideU32 = ((uint)(stride));
    if ((lid < strideU32)) {
      float otherNij = sumNij[(lid + strideU32)];
      float mineNij = sumNij[lid];
      sumNij[lid] = (mineNij + otherNij);
      float otherAi = sumAi[(lid + strideU32)];
      float mineAi = sumAi[lid];
      sumAi[lid] = (mineAi + otherAi);
      float otherBj = sumBj[(lid + strideU32)];
      float mineBj = sumBj[lid];
      sumBj[lid] = (mineBj + otherBj);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float sumNijFinal = sumNij[0u];
    float sumAiFinal = sumAi[0u];
    float sumBjFinal = sumBj[0u];
    float nM1 = (nVal - as_type<float>(0x3f800000u));
    float nChoose2 = ((nVal * nM1) / as_type<float>(0x40000000u));
    float expectedIdx = ((sumAiFinal * sumBjFinal) / nChoose2);
    float maxIdx = ((sumAiFinal + sumBjFinal) / as_type<float>(0x40000000u));
    float numerator = (sumNijFinal - expectedIdx);
    float denominator = (maxIdx - expectedIdx);
    bool denomIsZero = (denominator == as_type<float>(0x00000000u));
    float ariVal = ((denomIsZero) ? (as_type<float>(0x3f800000u)) : ((numerator / denominator)));
    ari[0u] = ariVal;
  }
}
