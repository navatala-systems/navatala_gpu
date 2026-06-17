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
extern "C" __global__ void navatala_dataframe_completeness_score_f32(const float* contingency, const float* rowSums, const float* colSums, const float* n, const unsigned int* numTrue, const unsigned int* numPred, float* completeness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float hcySum[256];
  __shared__ float hcSum[256];
  float nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  float localHCYAccum = __uint_as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int i = (idxU32 / numPredVal);
      float nij = contingency[idxU32];
      float ai = rowSums[i];
      bool nijGtZero = (nij > __uint_as_float(0x00000000u));
      bool aiGtZero = (ai > __uint_as_float(0x00000000u));
      bool bothGtZero = (nijGtZero && aiGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / ai);
        float logCondProb = log(condProb);
        float termHCY = (pij * logCondProb);
        float curHCY = localHCYAccum;
        float newHCY = (curHCY - termHCY);
        localHCYAccum = newHCY;
      }
    }
  }
  float localHCY = localHCYAccum;
  float localHCAccumComp = __uint_as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      float bj = colSums[jU32];
      bool bjGtZero = (bj > __uint_as_float(0x00000000u));
      if (bjGtZero) {
        float pj = (bj / nVal);
        float logPj = log(pj);
        float termHC = (pj * logPj);
        float curHC = localHCAccumComp;
        float newHC = (curHC - termHC);
        localHCAccumComp = newHC;
      }
    }
  }
  float localHC = localHCAccumComp;
  hcySum[lid] = localHCY;
  hcSum[lid] = localHC;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if ((lid < strideU32)) {
      float otherHCY = hcySum[(lid + strideU32)];
      float mineHCY = hcySum[lid];
      hcySum[lid] = (mineHCY + otherHCY);
      float otherHC = hcSum[(lid + strideU32)];
      float mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    __syncthreads();
  }
  if ((lid == 0u)) {
    float hcyFinal = hcySum[0u];
    float hcFinal = hcSum[0u];
    bool hcIsZero = (hcFinal == __uint_as_float(0x00000000u));
    float compVal = ((hcIsZero) ? (__uint_as_float(0x3f800000u)) : ((__uint_as_float(0x3f800000u) - (hcyFinal / hcFinal))));
    completeness[0u] = compVal;
  }
}
