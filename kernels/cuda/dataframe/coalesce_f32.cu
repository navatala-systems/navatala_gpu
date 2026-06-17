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
extern "C" __global__ void navatala_dataframe_coalesce_f32(const float* col1, const unsigned int* col1Valid, const float* col2, const unsigned int* col2Valid, const unsigned int* count, float* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int v1Word = col1Valid[wordIdx];
    unsigned int v1Bit = ((v1Word >> bitIdx) & 1u);
    bool col1IsValid = (v1Bit == 1u);
    unsigned int v2Word = col2Valid[wordIdx];
    unsigned int v2Bit = ((v2Word >> bitIdx) & 1u);
    bool col2IsValid = (v2Bit == 1u);
    float val1 = col1[gid];
    float val2 = col2[gid];
    float outVal = ((col1IsValid) ? (val1) : (val2));
    _output[gid] = outVal;
    bool eitherValid = (col1IsValid || col2IsValid);
    unsigned int outBit = ((eitherValid) ? ((1u << bitIdx)) : (0u));
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
