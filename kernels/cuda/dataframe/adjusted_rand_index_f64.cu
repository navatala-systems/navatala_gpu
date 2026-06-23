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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_adjusted_rand_index_f64(const double* contingency, const double* rowSums, const double* colSums, const double* n, const unsigned int* numTrue, const unsigned int* numPred, double* ari) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sumNij[256];
  __shared__ double sumAi[256];
  __shared__ double sumBj[256];
  double nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  double localSumNijAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      double nij = contingency[idxU32];
      double nijM1 = (nij - __longlong_as_double(0x3ff0000000000000ull));
      double choose2 = ((nij * nijM1) / __longlong_as_double(0x4000000000000000ull));
      double curSumNij = localSumNijAccum64;
      double newSumNij = (curSumNij + choose2);
      localSumNijAccum64 = newSumNij;
    }
  }
  double localSumNij = localSumNijAccum64;
  double localSumAiAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      double ai = rowSums[iU32];
      double aiM1 = (ai - __longlong_as_double(0x3ff0000000000000ull));
      double choose2Ai = ((ai * aiM1) / __longlong_as_double(0x4000000000000000ull));
      double curSumAi = localSumAiAccum64;
      double newSumAi = (curSumAi + choose2Ai);
      localSumAiAccum64 = newSumAi;
    }
  }
  double localSumAi = localSumAiAccum64;
  double localSumBjAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      double bj = colSums[jU32];
      double bjM1 = (bj - __longlong_as_double(0x3ff0000000000000ull));
      double choose2Bj = ((bj * bjM1) / __longlong_as_double(0x4000000000000000ull));
      double curSumBj = localSumBjAccum64;
      double newSumBj = (curSumBj + choose2Bj);
      localSumBjAccum64 = newSumBj;
    }
  }
  double localSumBj = localSumBjAccum64;
  sumNij[lid] = localSumNij;
  sumAi[lid] = localSumAi;
  sumBj[lid] = localSumBj;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if (lid < strideU32) {
      double otherNij = sumNij[(lid + strideU32)];
      double mineNij = sumNij[lid];
      sumNij[lid] = (mineNij + otherNij);
      double otherAi = sumAi[(lid + strideU32)];
      double mineAi = sumAi[lid];
      sumAi[lid] = (mineAi + otherAi);
      double otherBj = sumBj[(lid + strideU32)];
      double mineBj = sumBj[lid];
      sumBj[lid] = (mineBj + otherBj);
    }
    __syncthreads();
  }
  if (lid == 0u) {
    double sumNijFinal = sumNij[0u];
    double sumAiFinal = sumAi[0u];
    double sumBjFinal = sumBj[0u];
    double nM1 = (nVal - __longlong_as_double(0x3ff0000000000000ull));
    double nChoose2 = ((nVal * nM1) / __longlong_as_double(0x4000000000000000ull));
    double expectedIdx = ((sumAiFinal * sumBjFinal) / nChoose2);
    double maxIdx = ((sumAiFinal + sumBjFinal) / __longlong_as_double(0x4000000000000000ull));
    double numerator = (sumNijFinal - expectedIdx);
    double denominator = (maxIdx - expectedIdx);
    bool denomIsZero = (denominator == __longlong_as_double(0x0000000000000000ull));
    double ariVal = ((denomIsZero) ? (__longlong_as_double(0x3ff0000000000000ull)) : ((numerator / denominator)));
    ari[0u] = ariVal;
  }
}
