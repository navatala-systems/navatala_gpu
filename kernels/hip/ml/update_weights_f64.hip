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
extern "C" __global__ void navatala_ml_update_weights_f64(const double* responsibilities, const unsigned int* n, const unsigned int* k, double* weights) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numK = k[0];
  unsigned int j = gid;
  bool inBounds = (j < numK);
  if (inBounds) {
    double respSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int i = 0; i < (int)(numN); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int respIdx = ((iU32 * numK) + j);
      double r_ij = responsibilities[respIdx];
      double currRespSum = respSumAccum;
      double newRespSum = (currRespSum + r_ij);
      respSumAccum = newRespSum;
    }
    double respSum = respSumAccum;
    double nFloat = ((double)(numN));
    double weightVal = (respSum / nFloat);
    weights[j] = weightVal;
  }
}
