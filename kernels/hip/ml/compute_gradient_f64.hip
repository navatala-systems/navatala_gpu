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
extern "C" __global__ void navatala_ml_compute_gradient_f64(const double* P, const double* Q, const double* Y, const unsigned int* nPoints, const unsigned int* outputDims, double* grad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = outputDims[0];
  unsigned int totalGradElems = (n * d);
  bool inBounds = (gid < totalGradElems);
  if (inBounds) {
    unsigned int i = (gid / d);
    unsigned int k = (gid % d);
    double yi_k = Y[gid];
    double gradSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)(n); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      bool notDiag = (i != jU32);
      if (notDiag) {
        unsigned int pIdx = ((i * n) + jU32);
        double pij = P[pIdx];
        double qij = Q[pIdx];
        unsigned int yjIdx = ((jU32 * d) + k);
        double yj_k = Y[yjIdx];
        unsigned int baseI = (i * d);
        unsigned int baseJ = (jU32 * d);
        double distSqAccum = __longlong_as_double(0x0000000000000000ull);
        for (int m = 0; m < (int)(d); ++m) {
          unsigned int mU32 = ((unsigned int)(m));
          unsigned int idxIM = (baseI + mU32);
          unsigned int idxJM = (baseJ + mU32);
          double yi_m = Y[idxIM];
          double yj_m = Y[idxJM];
          double diffM = (yi_m - yj_m);
          double diffSqM = (diffM * diffM);
          double currentDistSq = distSqAccum;
          double newDistSq = (currentDistSq + diffSqM);
          distSqAccum = newDistSq;
        }
        double distSq = distSqAccum;
        double invDist = (__longlong_as_double(0x3ff0000000000000ull) / (__longlong_as_double(0x3ff0000000000000ull) + distSq));
        double diffK = (yi_k - yj_k);
        double pMinusQ = (pij - qij);
        double term = (pMinusQ * (diffK * invDist));
        double currentGradSum = gradSumAccum;
        double newGradSum = (currentGradSum + term);
        gradSumAccum = newGradSum;
      }
    }
    double gradSum = gradSumAccum;
    double gradVal = (__longlong_as_double(0x4010000000000000ull) * gradSum);
    grad[gid] = gradVal;
  }
}
