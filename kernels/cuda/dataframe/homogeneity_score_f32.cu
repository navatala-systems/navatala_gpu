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
extern "C" __global__ void navatala_dataframe_homogeneity_score_f32(const float* contingency, const float* rowSums, const float* colSums, const float* n, const unsigned int* numTrue, const unsigned int* numPred, float* homogeneity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float hycSum[256];
  __shared__ float hySum[256];
  float nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  float localHYCAccum = __uint_as_float(0x00000000u);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int j = (idxU32 % numPredVal);
      float nij = contingency[idxU32];
      float bj = colSums[j];
      bool nijGtZero = (nij > __uint_as_float(0x00000000u));
      bool bjGtZero = (bj > __uint_as_float(0x00000000u));
      bool bothGtZero = (nijGtZero && bjGtZero);
      if (bothGtZero) {
        float pij = (nij / nVal);
        float condProb = (nij / bj);
        float logCondProb = log(condProb);
        float termHYC = (pij * logCondProb);
        float curHYC = localHYCAccum;
        float newHYC = (curHYC - termHYC);
        localHYCAccum = newHYC;
      }
    }
  }
  float localHYC = localHYCAccum;
  float localHYAccumHom = __uint_as_float(0x00000000u);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      float ai = rowSums[iU32];
      bool aiGtZero = (ai > __uint_as_float(0x00000000u));
      if (aiGtZero) {
        float pi = (ai / nVal);
        float logPi = log(pi);
        float termHY = (pi * logPi);
        float curHY = localHYAccumHom;
        float newHY = (curHY - termHY);
        localHYAccumHom = newHY;
      }
    }
  }
  float localHY = localHYAccumHom;
  hycSum[lid] = localHYC;
  hySum[lid] = localHY;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if ((lid < strideU32)) {
      float otherHYC = hycSum[(lid + strideU32)];
      float mineHYC = hycSum[lid];
      hycSum[lid] = (mineHYC + otherHYC);
      float otherHY = hySum[(lid + strideU32)];
      float mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
    }
    __syncthreads();
  }
  if ((lid == 0u)) {
    float hycFinal = hycSum[0u];
    float hyFinal = hySum[0u];
    bool hyIsZero = (hyFinal == __uint_as_float(0x00000000u));
    float homoVal = ((hyIsZero) ? (__uint_as_float(0x3f800000u)) : ((__uint_as_float(0x3f800000u) - (hycFinal / hyFinal))));
    homogeneity[0u] = homoVal;
  }
}
