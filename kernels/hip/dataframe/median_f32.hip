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
extern "C" __global__ void navatala_dataframe_median_f32(const float* sortedInput, const unsigned int* count, float* median) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gidU32 = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gidU32 == 0u)) {
    unsigned int n = count[0u];
    unsigned int nMinus1 = (n - 1u);
    float nMinus1Float = ((float)(nMinus1));
    float idx = (__uint_as_float(0x3f000000u) * nMinus1Float);
    float lowerFloat = floor(idx);
    unsigned int lower = ((unsigned int)(lowerFloat));
    unsigned int upperCandidate = (lower + 1u);
    unsigned int upper = (((upperCandidate < n)) ? (upperCandidate) : (nMinus1));
    float frac = (idx - lowerFloat);
    float valLower = sortedInput[lower];
    float valUpper = sortedInput[upper];
    float oneMinusFrac = (__uint_as_float(0x3f800000u) - frac);
    float termLower = (oneMinusFrac * valLower);
    float termUpper = (frac * valUpper);
    float medianVal = (termLower + termUpper);
    median[0u] = medianVal;
  }
}
