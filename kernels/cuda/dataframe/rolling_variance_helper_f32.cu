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
extern "C" __global__ void navatala_dataframe_rolling_variance_helper_f32(const float* data, const unsigned int* validity, const unsigned int* windowSize, const unsigned int* count, long long* sumOutput, long long* sumSqOutput, unsigned int* countOutput, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    float val = data[gid];
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    float scaleFactor = __uint_as_float(0x47800000u);
    float scaledVal = (val * scaleFactor);
    long long scaledInt = ((long long)(scaledVal));
    long long sumVal = ((isValid) ? (scaledInt) : (0));
    float valSq = (val * val);
    float scaledSq = (valSq * scaleFactor);
    long long scaledSqInt = ((long long)(scaledSq));
    long long sumSqVal = ((isValid) ? (scaledSqInt) : (0));
    unsigned int cntVal = ((isValid) ? (1u) : (0u));
    sumOutput[gid] = sumVal;
    sumSqOutput[gid] = sumSqVal;
    countOutput[gid] = cntVal;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
