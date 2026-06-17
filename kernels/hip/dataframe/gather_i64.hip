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
extern "C" __global__ void navatala_dataframe_gather_i64(const long long* _input, const unsigned int* inputValid, const int* indices, const unsigned int* inputSize, const unsigned int* outputSize, long long* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nIn = inputSize[0u];
  unsigned int nOut = outputSize[0u];
  bool inBounds = (gid < nOut);
  if (inBounds) {
    int idx = indices[gid];
    unsigned int idxU32 = ((unsigned int)(idx));
    bool idxValid = ((idx >= 0) && (idxU32 < nIn));
    unsigned int srcWordIdx = (idxU32 / 32u);
    unsigned int srcBitIdx = (idxU32 % 32u);
    unsigned int srcValidWord = ((idxValid) ? (inputValid[srcWordIdx]) : (0u));
    unsigned int srcValidBit = ((srcValidWord >> srcBitIdx) & 1u);
    bool srcIsValid = (srcValidBit == 1u);
    bool outIsValid = (idxValid && srcIsValid);
    long long val = ((outIsValid) ? (_input[idxU32]) : (0));
    _output[gid] = val;
    unsigned int outWordIdx = (gid / 32u);
    unsigned int outBitIdx = (gid % 32u);
    unsigned int outBit = ((outIsValid) ? ((1u << outBitIdx)) : (0u));
    atomicOr(&outputValid[outWordIdx], outBit);
  }
}
