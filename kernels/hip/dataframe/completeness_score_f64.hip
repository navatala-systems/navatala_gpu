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
extern "C" __global__ void navatala_dataframe_completeness_score_f64(const double* contingency, const double* rowSums, const double* colSums, const double* n, const unsigned int* numTrue, const unsigned int* numPred, double* completeness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double hcySum[256];
  __shared__ double hcSum[256];
  double nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  double localHCYAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int i = (idxU32 / numPredVal);
      double nij = contingency[idxU32];
      double ai = rowSums[i];
      bool nijGtZero = (nij > __longlong_as_double(0x0000000000000000ull));
      bool aiGtZero = (ai > __longlong_as_double(0x0000000000000000ull));
      bool bothGtZero = (nijGtZero && aiGtZero);
      if (bothGtZero) {
        double pij = (nij / nVal);
        double condProb = (nij / ai);
        double logCondProb = log(condProb);
        double termHCY = (pij * logCondProb);
        double curHCY = localHCYAccum64;
        double newHCY = (curHCY - termHCY);
        localHCYAccum64 = newHCY;
      }
    }
  }
  double localHCY = localHCYAccum64;
  double localHCAccumComp64 = __longlong_as_double(0x0000000000000000ull);
  for (int j = 0; j < (int)(numPredVal); ++j) {
    unsigned int jU32 = ((unsigned int)(j));
    bool shouldProcess = ((jU32 % 256u) == lid);
    if (shouldProcess) {
      double bj = colSums[jU32];
      bool bjGtZero = (bj > __longlong_as_double(0x0000000000000000ull));
      if (bjGtZero) {
        double pj = (bj / nVal);
        double logPj = log(pj);
        double termHC = (pj * logPj);
        double curHC = localHCAccumComp64;
        double newHC = (curHC - termHC);
        localHCAccumComp64 = newHC;
      }
    }
  }
  double localHC = localHCAccumComp64;
  hcySum[lid] = localHCY;
  hcSum[lid] = localHC;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if (lid < strideU32) {
      double otherHCY = hcySum[(lid + strideU32)];
      double mineHCY = hcySum[lid];
      hcySum[lid] = (mineHCY + otherHCY);
      double otherHC = hcSum[(lid + strideU32)];
      double mineHC = hcSum[lid];
      hcSum[lid] = (mineHC + otherHC);
    }
    __syncthreads();
  }
  if (lid == 0u) {
    double hcyFinal = hcySum[0u];
    double hcFinal = hcSum[0u];
    bool hcIsZero = (hcFinal == __longlong_as_double(0x0000000000000000ull));
    double compVal = ((hcIsZero) ? (__longlong_as_double(0x3ff0000000000000ull)) : ((__longlong_as_double(0x3ff0000000000000ull) - (hcyFinal / hcFinal))));
    completeness[0u] = compVal;
  }
}
