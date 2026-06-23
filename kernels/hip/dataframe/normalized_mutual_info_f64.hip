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
extern "C" __global__ void navatala_dataframe_normalized_mutual_info_f64(const double* contingency, const double* rowSums, const double* colSums, const double* n, const unsigned int* numTrue, const unsigned int* numPred, double* nmi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double miSum[256];
  __shared__ double hySum[256];
  __shared__ double hcSum[256];
  double nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  double localMIAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int i = (idxU32 / numPredVal);
      unsigned int j = (idxU32 % numPredVal);
      double nij = contingency[idxU32];
      double ai = rowSums[i];
      double bj = colSums[j];
      bool nijGtZero = (nij > __longlong_as_double(0x0000000000000000ull));
      if (nijGtZero) {
        double pij = (nij / nVal);
        double aibj = (ai * bj);
        double ratio = ((nVal * nij) / aibj);
        double logRatio = log(ratio);
        double termMI = (pij * logRatio);
        double curMI = localMIAccum64;
        double newMI = (curMI + termMI);
        localMIAccum64 = newMI;
      }
    }
  }
  double localMI = localMIAccum64;
  double localHYAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      double aiHY = rowSums[iU32];
      bool aiGtZero = (aiHY > __longlong_as_double(0x0000000000000000ull));
      if (aiGtZero) {
        double pi = (aiHY / nVal);
        double logPi = log(pi);
        double termHY = (pi * logPi);
        double curHY = localHYAccum64;
        double newHY = (curHY - termHY);
        localHYAccum64 = newHY;
      }
    }
  }
  double localHY = localHYAccum64;
  double localHCAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      double bjHC = colSums[jU32];
      bool bjGtZero = (bjHC > __longlong_as_double(0x0000000000000000ull));
      if (bjGtZero) {
        double pj = (bjHC / nVal);
        double logPj = log(pj);
        double termHC = (pj * logPj);
        double curHC = localHCAccum64;
        double newHC = (curHC - termHC);
        localHCAccum64 = newHC;
      }
    }
  }
  double localHC = localHCAccum64;
  miSum[lid] = localMI;
  hySum[lid] = localHY;
  hcSum[lid] = localHC;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if (lid < strideU32) {
      double otherMI = miSum[(lid + strideU32)];
      double mineMI = miSum[lid];
      miSum[lid] = (mineMI + otherMI);
      double otherHY = hySum[(lid + strideU32)];
      double mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
      double otherHC = hcSum[(lid + strideU32)];
      double mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    __syncthreads();
  }
  if (lid == 0u) {
    double miFinal = miSum[0u];
    double hyFinal = hySum[0u];
    double hcFinal = hcSum[0u];
    double hSum = (hyFinal + hcFinal);
    bool hSumIsZero = (hSum == __longlong_as_double(0x0000000000000000ull));
    double nmiVal = ((hSumIsZero) ? (__longlong_as_double(0x3ff0000000000000ull)) : (((__longlong_as_double(0x4000000000000000ull) * miFinal) / hSum)));
    nmi[0u] = nmiVal;
  }
}
