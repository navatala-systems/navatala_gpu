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
extern "C" __global__ void navatala_dataframe_iqr_f64(const double* sortedInput, const unsigned int* count, double* iqr) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gidU32 = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gidU32 == 0u)) {
    unsigned int n = count[0u];
    unsigned int nMinus1 = (n - 1u);
    double nMinus1Float = ((double)(nMinus1));
    double idxQ1 = (__longlong_as_double(0x3fd0000000000000ull) * nMinus1Float);
    double lowerQ1Float = floor(idxQ1);
    unsigned int lowerQ1 = ((unsigned int)(lowerQ1Float));
    unsigned int upperQ1Candidate = (lowerQ1 + 1u);
    unsigned int upperQ1 = (((upperQ1Candidate < n)) ? (upperQ1Candidate) : (nMinus1));
    double fracQ1 = (idxQ1 - lowerQ1Float);
    double valLowerQ1 = sortedInput[lowerQ1];
    double valUpperQ1 = sortedInput[upperQ1];
    double oneMinusFracQ1 = (__longlong_as_double(0x3ff0000000000000ull) - fracQ1);
    double q1 = ((oneMinusFracQ1 * valLowerQ1) + (fracQ1 * valUpperQ1));
    double idxQ3 = (__longlong_as_double(0x3fe8000000000000ull) * nMinus1Float);
    double lowerQ3Float = floor(idxQ3);
    unsigned int lowerQ3 = ((unsigned int)(lowerQ3Float));
    unsigned int upperQ3Candidate = (lowerQ3 + 1u);
    unsigned int upperQ3 = (((upperQ3Candidate < n)) ? (upperQ3Candidate) : (nMinus1));
    double fracQ3 = (idxQ3 - lowerQ3Float);
    double valLowerQ3 = sortedInput[lowerQ3];
    double valUpperQ3 = sortedInput[upperQ3];
    double oneMinusFracQ3 = (__longlong_as_double(0x3ff0000000000000ull) - fracQ3);
    double q3 = ((oneMinusFracQ3 * valLowerQ3) + (fracQ3 * valUpperQ3));
    double iqrVal = (q3 - q1);
    iqr[0u] = iqrVal;
  }
}
