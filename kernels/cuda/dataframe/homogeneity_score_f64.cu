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
extern "C" __global__ void navatala_dataframe_homogeneity_score_f64(const double* contingency, const double* rowSums, const double* colSums, const double* n, const unsigned int* numTrue, const unsigned int* numPred, double* homogeneity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double hycSum[256];
  __shared__ double hySum[256];
  double nVal = n[0u];
  unsigned int numTrueVal = numTrue[0u];
  unsigned int numPredVal = numPred[0u];
  unsigned int totalCells = (numTrueVal * numPredVal);
  double localHYCAccum64 = __longlong_as_double(0x0000000000000000ull);
  for (int idx = 0; idx < (int)(totalCells); ++idx) {
    unsigned int idxU32 = ((unsigned int)(idx));
    bool shouldProcess = ((idxU32 % 256u) == lid);
    if (shouldProcess) {
      unsigned int j = (idxU32 % numPredVal);
      double nij = contingency[idxU32];
      double bj = colSums[j];
      bool nijGtZero = (nij > __longlong_as_double(0x0000000000000000ull));
      bool bjGtZero = (bj > __longlong_as_double(0x0000000000000000ull));
      bool bothGtZero = (nijGtZero && bjGtZero);
      if (bothGtZero) {
        double pij = (nij / nVal);
        double condProb = (nij / bj);
        double logCondProb = log(condProb);
        double termHYC = (pij * logCondProb);
        double curHYC = localHYCAccum64;
        double newHYC = (curHYC - termHYC);
        localHYCAccum64 = newHYC;
      }
    }
  }
  double localHYC = localHYCAccum64;
  double localHYAccumHom64 = __longlong_as_double(0x0000000000000000ull);
  for (int i = 0; i < (int)(numTrueVal); ++i) {
    unsigned int iU32 = ((unsigned int)(i));
    bool shouldProcess = ((iU32 % 256u) == lid);
    if (shouldProcess) {
      double ai = rowSums[iU32];
      bool aiGtZero = (ai > __longlong_as_double(0x0000000000000000ull));
      if (aiGtZero) {
        double pi = (ai / nVal);
        double logPi = log(pi);
        double termHY = (pi * logPi);
        double curHY = localHYAccumHom64;
        double newHY = (curHY - termHY);
        localHYAccumHom64 = newHY;
      }
    }
  }
  double localHY = localHYAccumHom64;
  hycSum[lid] = localHYC;
  hySum[lid] = localHY;
  __syncthreads();
  for (int stride = 0; stride < (int)(128u); ++stride) {
    unsigned int strideU32 = ((unsigned int)(stride));
    if ((lid < strideU32)) {
      double otherHYC = hycSum[(lid + strideU32)];
      double mineHYC = hycSum[lid];
      hycSum[lid] = (mineHYC + otherHYC);
      double otherHY = hySum[(lid + strideU32)];
      double mineHY = hySum[lid];
      hySum[lid] = (mineHY + otherHY);
    }
    __syncthreads();
  }
  if ((lid == 0u)) {
    double hycFinal = hycSum[0u];
    double hyFinal = hySum[0u];
    bool hyIsZero = (hyFinal == __longlong_as_double(0x0000000000000000ull));
    double homoVal = ((hyIsZero) ? (__longlong_as_double(0x3ff0000000000000ull)) : ((__longlong_as_double(0x3ff0000000000000ull) - (hycFinal / hyFinal))));
    homogeneity[0u] = homoVal;
  }
}
