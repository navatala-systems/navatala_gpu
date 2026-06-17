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
extern "C" __global__ void navatala_ml_compute_covariance_matrix_f64(const double* Xc, const unsigned int* nSamples, const unsigned int* nFeatures, double* Cov) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int covSize = (d * d);
  bool inBounds = (gid < covSize);
  if (inBounds) {
    unsigned int i = (gid / d);
    unsigned int j = (gid % d);
    double sum = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(n); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int idxI = ((kU32 * d) + i);
      unsigned int idxJ = ((kU32 * d) + j);
      double xi = Xc[idxI];
      double xj = Xc[idxJ];
      double prod = (xi * xj);
      double currSum = sum;
      double newSum = (currSum + prod);
      sum = newSum;
    }
    double finalSum = sum;
    unsigned int nMinus1 = (n - 1u);
    double divisor = ((double)(nMinus1));
    double covVal = (finalSum / divisor);
    Cov[gid] = covVal;
  }
}
