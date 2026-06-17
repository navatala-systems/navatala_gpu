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
extern "C" __global__ void navatala_dataframe_normalized_mutual_info_f32(const float* contingency, const float* rowSums, const float* colSums, const float* n, const unsigned int* numTrue, const unsigned int* numPred, float* nmi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float miSum[256];
  __shared__ float hySum[256];
  __shared__ float hcSum[256];
  float nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  float localMIAccum = __uint_as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int i = (idxU32 / numPredVal);
      unsigned int j = (idxU32 % numPredVal);
      float nij = contingency[idxU32];
      float ai = rowSums[i];
      float bj = colSums[j];
      bool nijGtZero = (nij > __uint_as_float(0x00000000u));
      if (nijGtZero) {
        float pij = (nij / nVal);
        float aibj = (ai * bj);
        float ratio = ((nVal * nij) / aibj);
        float logRatio = log(ratio);
        float termMI = (pij * logRatio);
        float curMI = localMIAccum;
        float newMI = (curMI + termMI);
        localMIAccum = newMI;
      }
    }
  }
  float localMI = localMIAccum;
  float localHYAccum = __uint_as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      float aiHY = rowSums[iU32];
      bool aiGtZero = (aiHY > __uint_as_float(0x00000000u));
      if (aiGtZero) {
        float pi = (aiHY / nVal);
        float logPi = log(pi);
        float termHY = (pi * logPi);
        float curHY = localHYAccum;
        float newHY = (curHY - termHY);
        localHYAccum = newHY;
      }
    }
  }
  float localHY = localHYAccum;
  float localHCAccum = __uint_as_float(0x00000000u);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      float bjHC = colSums[jU32];
      bool bjGtZero = (bjHC > __uint_as_float(0x00000000u));
      if (bjGtZero) {
        float pj = (bjHC / nVal);
        float logPj = log(pj);
        float termHC = (pj * logPj);
        float curHC = localHCAccum;
        float newHC = (curHC - termHC);
        localHCAccum = newHC;
      }
    }
  }
  float localHC = localHCAccum;
  miSum[lid] = localMI;
  hySum[lid] = localHY;
  hcSum[lid] = localHC;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if ((lid < strideU32)) {
      float otherMI = miSum[(lid + strideU32)];
      float mineMI = miSum[lid];
      miSum[lid] = (mineMI + otherMI);
      float otherHY = hySum[(lid + strideU32)];
      float mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
      float otherHC = hcSum[(lid + strideU32)];
      float mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    __syncthreads();
  }
  if ((lid == 0u)) {
    float miFinal = miSum[0u];
    float hyFinal = hySum[0u];
    float hcFinal = hcSum[0u];
    float hSum = (hyFinal + hcFinal);
    bool hSumIsZero = (hSum == __uint_as_float(0x00000000u));
    float nmiVal = ((hSumIsZero) ? (__uint_as_float(0x3f800000u)) : (((__uint_as_float(0x40000000u) * miFinal) / hSum)));
    nmi[0u] = nmiVal;
  }
}
