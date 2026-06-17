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
extern "C" __global__ void navatala_dataframe_subtract_timestamps(const long long* timestamps1, const long long* timestamps2, const unsigned int* validity1, const unsigned int* validity2, const unsigned int* count, long long* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord1 = validity1[wordIdx];
    unsigned int validBit1 = ((validWord1 >> bitIdx) & 1u);
    unsigned int validWord2 = validity2[wordIdx];
    unsigned int validBit2 = ((validWord2 >> bitIdx) & 1u);
    unsigned int bothValid = (validBit1 & validBit2);
    bool isValid = (bothValid == 1u);
    long long ts1 = timestamps1[gid];
    long long ts2 = timestamps2[gid];
    long long diff = (ts2 - ts1);
    long long outVal = ((isValid) ? (diff) : (0));
    _output[gid] = outVal;
    unsigned int outBit = (bothValid << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
