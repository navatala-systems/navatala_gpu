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

__kernel void navatala_dataframe_adjusted_rand_index_f32(__global const float* contingency, __global const float* rowSums, __global const float* colSums, __global const float* n, __global const uint* numTrue, __global const uint* numPred, __global float* ari) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sumNij[256];
  __local float sumAi[256];
  __local float sumBj[256];
  float nVal = n[(uint)(0u)];
  uint numTrueVal = numTrue[(uint)(0u)];
  uint numPredVal = numPred[(uint)(0u)];
  uint totalCells = (numTrueVal * numPredVal);
  float localSumNijAccum = as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    uint idxU32 = ((uint)(idx));
    bool shouldProcess = ((idxU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float nij = contingency[idxU32];
      float nijM1 = (nij - as_float(0x3f800000u));
      float choose2 = ((nij * nijM1) / as_float(0x40000000u));
      float curSumNij = localSumNijAccum;
      float newSumNij = (curSumNij + choose2);
      localSumNijAccum = newSumNij;
    }
  }
  float localSumNij = localSumNijAccum;
  float localSumAiAccum = as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    uint iU32 = ((uint)(i));
    bool shouldProcess = ((iU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      float aiM1 = (ai - as_float(0x3f800000u));
      float choose2Ai = ((ai * aiM1) / as_float(0x40000000u));
      float curSumAi = localSumAiAccum;
      float newSumAi = (curSumAi + choose2Ai);
      localSumAiAccum = newSumAi;
    }
  }
  float localSumAi = localSumAiAccum;
  float localSumBjAccum = as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    uint jU32 = ((uint)(j));
    bool shouldProcess = ((jU32 % (uint)(256u)) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      float bjM1 = (bj - as_float(0x3f800000u));
      float choose2Bj = ((bj * bjM1) / as_float(0x40000000u));
      float curSumBj = localSumBjAccum;
      float newSumBj = (curSumBj + choose2Bj);
      localSumBjAccum = newSumBj;
    }
  }
  float localSumBj = localSumBjAccum;
  sumNij[lid] = localSumNij;
  sumAi[lid] = localSumAi;
  sumBj[lid] = localSumBj;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = 0; stride < (int)((uint)(128u)); ++stride) {
    uint strideU32 = ((uint)(stride));
    if (lid < strideU32) {
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float sumNijFinal = sumNij[(uint)(0u)];
    float sumAiFinal = sumAi[(uint)(0u)];
    float sumBjFinal = sumBj[(uint)(0u)];
    float nM1 = (nVal - as_float(0x3f800000u));
    float nChoose2 = ((nVal * nM1) / as_float(0x40000000u));
    float expectedIdx = ((sumAiFinal * sumBjFinal) / nChoose2);
    float maxIdx = ((sumAiFinal + sumBjFinal) / as_float(0x40000000u));
    float numerator = (sumNijFinal - expectedIdx);
    float denominator = (maxIdx - expectedIdx);
    bool denomIsZero = (denominator == as_float(0x00000000u));
    float ariVal = ((denomIsZero) ? (as_float(0x3f800000u)) : ((numerator / denominator)));
    ari[(uint)(0u)] = ariVal;
  }
}
