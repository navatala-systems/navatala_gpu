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
extern "C" __global__ void navatala_ml_compute_pairwise_affinity_f64(const double* X, const double* sigma, const unsigned int* nPoints, const unsigned int* nDims, double* P) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = nDims[0];
  unsigned int totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    unsigned int i = (gid / n);
    unsigned int j = (gid % n);
    double sigmaI = sigma[i];
    double twoSigmaSq = (__longlong_as_double(0x4000000000000000ull) * (sigmaI * sigmaI));
    unsigned int baseI = (i * d);
    unsigned int baseJ = (j * d);
    double distSqAccum = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int idxI = (baseI + kU32);
      unsigned int idxJ = (baseJ + kU32);
      double xi_k = X[idxI];
      double xj_k = X[idxJ];
      double diff = (xi_k - xj_k);
      double diffSq = (diff * diff);
      double currentDistSq = distSqAccum;
      double newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    double distSq = distSqAccum;
    bool isDiag = (i == j);
    double negDistOverSigma = ((__longlong_as_double(0x0000000000000000ull) - distSq) / twoSigmaSq);
    double pVal = exp(negDistOverSigma);
    if (isDiag) {
      P[gid] = __longlong_as_double(0x0000000000000000ull);
    } else {
      P[gid] = pVal;
    }
  }
}
