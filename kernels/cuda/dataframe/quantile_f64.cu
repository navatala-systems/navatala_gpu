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
extern "C" __global__ void navatala_dataframe_quantile_f64(const double* sortedInput, const unsigned int* count, const double* quantileP, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gidU32 = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gidU32 == 0u)) {
    unsigned int n = count[0u];
    double p = quantileP[0u];
    double pClamped = (((p < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((p > __longlong_as_double(0x3ff0000000000000ull))) ? (__longlong_as_double(0x3ff0000000000000ull)) : (p))));
    unsigned int nMinus1 = (n - 1u);
    double nMinus1Float = ((double)(nMinus1));
    double idx = (pClamped * nMinus1Float);
    double lowerFloat = floor(idx);
    unsigned int lower = ((unsigned int)(lowerFloat));
    unsigned int upperCandidate = (lower + 1u);
    unsigned int upper = (((upperCandidate < n)) ? (upperCandidate) : (nMinus1));
    double frac = (idx - lowerFloat);
    double valLower = sortedInput[lower];
    double valUpper = sortedInput[upper];
    double oneMinusFrac = (__longlong_as_double(0x3ff0000000000000ull) - frac);
    double termLower = (oneMinusFrac * valLower);
    double termUpper = (frac * valUpper);
    double quantileVal = (termLower + termUpper);
    result[0u] = quantileVal;
  }
}
