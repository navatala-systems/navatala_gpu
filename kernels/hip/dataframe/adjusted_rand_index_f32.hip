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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_adjusted_rand_index_f32(const float* contingency, const float* rowSums, const float* colSums, const float* n, const unsigned int* numTrue, const unsigned int* numPred, float* ari) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sumNij[256];
  __shared__ float sumAi[256];
  __shared__ float sumBj[256];
  float nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  float localSumNijAccum = __uint_as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      float nij = contingency[idxU32];
      float nijM1 = (nij - __uint_as_float(0x3f800000u));
      float choose2 = ((nij * nijM1) / __uint_as_float(0x40000000u));
      float curSumNij = localSumNijAccum;
      float newSumNij = (curSumNij + choose2);
      localSumNijAccum = newSumNij;
    }
  }
  float localSumNij = localSumNijAccum;
  float localSumAiAccum = __uint_as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      float aiM1 = (ai - __uint_as_float(0x3f800000u));
      float choose2Ai = ((ai * aiM1) / __uint_as_float(0x40000000u));
      float curSumAi = localSumAiAccum;
      float newSumAi = (curSumAi + choose2Ai);
      localSumAiAccum = newSumAi;
    }
  }
  float localSumAi = localSumAiAccum;
  float localSumBjAccum = __uint_as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      float bjM1 = (bj - __uint_as_float(0x3f800000u));
      float choose2Bj = ((bj * bjM1) / __uint_as_float(0x40000000u));
      float curSumBj = localSumBjAccum;
      float newSumBj = (curSumBj + choose2Bj);
      localSumBjAccum = newSumBj;
    }
  }
  float localSumBj = localSumBjAccum;
  sumNij[lid] = localSumNij;
  sumAi[lid] = localSumAi;
  sumBj[lid] = localSumBj;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
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
    __syncthreads();
  }
  if (lid == 0u) {
    float sumNijFinal = sumNij[0u];
    float sumAiFinal = sumAi[0u];
    float sumBjFinal = sumBj[0u];
    float nM1 = (nVal - __uint_as_float(0x3f800000u));
    float nChoose2 = ((nVal * nM1) / __uint_as_float(0x40000000u));
    float expectedIdx = ((sumAiFinal * sumBjFinal) / nChoose2);
    float maxIdx = ((sumAiFinal + sumBjFinal) / __uint_as_float(0x40000000u));
    float numerator = (sumNijFinal - expectedIdx);
    float denominator = (maxIdx - expectedIdx);
    bool denomIsZero = (denominator == __uint_as_float(0x00000000u));
    float ariVal = ((denomIsZero) ? (__uint_as_float(0x3f800000u)) : ((numerator / denominator)));
    ari[0u] = ariVal;
  }
}
