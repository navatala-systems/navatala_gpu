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
extern "C" __global__ void navatala_ml_compute_neighbor_counts_f64(const double* points, const double* epsilon, const unsigned int* nPoints, const unsigned int* nDims, unsigned int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = nDims[0];
  double eps = epsilon[0];
  double epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int iBase = (gid * d);
    unsigned int countAccum = 0u;
    for (int j = 0; j < (int)(n); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int jBase = (jU32 * d);
      double distSqAccum = __longlong_as_double(0x0000000000000000ull);
      for (int k = 0; k < (int)(d); ++k) {
        unsigned int kU32 = ((unsigned int)(k));
        unsigned int iIdx = (iBase + kU32);
        unsigned int jIdx = (jBase + kU32);
        double iVal = points[iIdx];
        double jVal = points[jIdx];
        double diff = (iVal - jVal);
        double diffSq = (diff * diff);
        double currentDistSq = distSqAccum;
        double newDistSq = (currentDistSq + diffSq);
        distSqAccum = newDistSq;
      }
      double finalDistSq = distSqAccum;
      if ((finalDistSq < epsSq)) {
        unsigned int currentCount = countAccum;
        unsigned int newCount = (currentCount + 1u);
        countAccum = newCount;
      }
    }
    unsigned int finalCount = countAccum;
    counts[gid] = finalCount;
  }
}
