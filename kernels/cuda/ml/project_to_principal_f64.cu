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
extern "C" __global__ void navatala_ml_project_to_principal_f64(const double* X, const double* V, const unsigned int* nSamples, const unsigned int* nFeatures, const unsigned int* nComponents, double* Z) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int kComp = nComponents[0];
  unsigned int totalOutput = (n * kComp);
  bool inBounds = (gid < totalOutput);
  if (inBounds) {
    unsigned int i = (gid / kComp);
    unsigned int c = (gid % kComp);
    double sum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)(d); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int xIdx = ((i * d) + jU32);
      unsigned int vIdx = ((c * d) + jU32);
      double xVal = X[xIdx];
      double vVal = V[vIdx];
      double prod = (xVal * vVal);
      double currSum = sum;
      double newSum = (currSum + prod);
      sum = newSum;
    }
    double finalSum = sum;
    Z[gid] = finalSum;
  }
}
