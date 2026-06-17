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
extern "C" __global__ void navatala_dataframe_rolling_sum_f32(const float* data, const unsigned int* validity, const unsigned int* windowSize, const unsigned int* minPeriods, const unsigned int* count, float* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int w = windowSize[0u];
  unsigned int minP = minPeriods[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int windowStartRaw = (((int)(gid)) - (((int)(w)) - 1));
    int windowStartClamped = (((windowStartRaw < 0)) ? (0) : (windowStartRaw));
    unsigned int windowStart = ((unsigned int)(windowStartClamped));
    float val = data[gid];
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    float outVal = ((isValid) ? (val) : (__uint_as_float(0x00000000u)));
    _output[gid] = outVal;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
