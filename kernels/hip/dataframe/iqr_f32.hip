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
extern "C" __global__ void navatala_dataframe_iqr_f32(const float* sortedInput, const unsigned int* count, float* iqr) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gidU32 = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gidU32 == 0u)) {
    unsigned int n = count[0u];
    unsigned int nMinus1 = (n - 1u);
    float nMinus1Float = ((float)(nMinus1));
    float idxQ1 = (__uint_as_float(0x3e800000u) * nMinus1Float);
    float lowerQ1Float = floor(idxQ1);
    unsigned int lowerQ1 = ((unsigned int)(lowerQ1Float));
    unsigned int upperQ1Candidate = (lowerQ1 + 1u);
    unsigned int upperQ1 = (((upperQ1Candidate < n)) ? (upperQ1Candidate) : (nMinus1));
    float fracQ1 = (idxQ1 - lowerQ1Float);
    float valLowerQ1 = sortedInput[lowerQ1];
    float valUpperQ1 = sortedInput[upperQ1];
    float oneMinusFracQ1 = (__uint_as_float(0x3f800000u) - fracQ1);
    float q1 = ((oneMinusFracQ1 * valLowerQ1) + (fracQ1 * valUpperQ1));
    float idxQ3 = (__uint_as_float(0x3f400000u) * nMinus1Float);
    float lowerQ3Float = floor(idxQ3);
    unsigned int lowerQ3 = ((unsigned int)(lowerQ3Float));
    unsigned int upperQ3Candidate = (lowerQ3 + 1u);
    unsigned int upperQ3 = (((upperQ3Candidate < n)) ? (upperQ3Candidate) : (nMinus1));
    float fracQ3 = (idxQ3 - lowerQ3Float);
    float valLowerQ3 = sortedInput[lowerQ3];
    float valUpperQ3 = sortedInput[upperQ3];
    float oneMinusFracQ3 = (__uint_as_float(0x3f800000u) - fracQ3);
    float q3 = ((oneMinusFracQ3 * valLowerQ3) + (fracQ3 * valUpperQ3));
    float iqrVal = (q3 - q1);
    iqr[0u] = iqrVal;
  }
}
