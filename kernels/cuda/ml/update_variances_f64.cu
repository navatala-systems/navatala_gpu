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
extern "C" __global__ void navatala_ml_update_variances_f64(const double* data, const double* means, const double* responsibilities, const unsigned int* n, const unsigned int* k, const unsigned int* d, double* variances) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numK = k[0];
  unsigned int numD = d[0];
  unsigned int kd = (numK * numD);
  bool inBounds = (gid < kd);
  if (inBounds) {
    unsigned int j = (gid / numD);
    unsigned int l = (gid % numD);
    double mu_jl = means[gid];
    double weightedSqSumAccum = __longlong_as_double(0x0000000000000000ull);
    double respSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(numN); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int respIdx = ((iU32 * numK) + j);
      double r_ij = responsibilities[respIdx];
      unsigned int dataIdx = ((iU32 * numD) + l);
      double x_il = data[dataIdx];
      double diff = (x_il - mu_jl);
      double diffSq = (diff * diff);
      double weightedSq = (r_ij * diffSq);
      double currWeightedSqSum = weightedSqSumAccum;
      double newWeightedSqSum = (currWeightedSqSum + weightedSq);
      weightedSqSumAccum = newWeightedSqSum;
      double currRespSum = respSumAccum;
      double newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    double weightedSqSum = weightedSqSumAccum;
    double respSum = respSumAccum;
    double rawVar = (weightedSqSum / respSum);
    double varVal = (rawVar + __longlong_as_double(0x3eb0c6f7a0b5ed8dull));
    variances[gid] = varVal;
  }
}
