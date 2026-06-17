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
extern "C" __global__ void navatala_dataframe_gather_concat_i64(const long long* src, const unsigned int* srcValid, const unsigned int* srcLen, const unsigned int* destOffset, long long* dest, unsigned int* destValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = srcLen[0u];
  unsigned int offset = destOffset[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    long long val = src[gid];
    unsigned int destIdx = (offset + gid);
    dest[destIdx] = val;
    unsigned int srcWordIdx = (gid / 32u);
    unsigned int srcBitIdx = (gid % 32u);
    unsigned int srcValidWord = srcValid[srcWordIdx];
    unsigned int srcValidBit = ((srcValidWord >> srcBitIdx) & 1u);
    unsigned int destWordIdx = (destIdx / 32u);
    unsigned int destBitIdx = (destIdx % 32u);
    unsigned int outBit = (srcValidBit << destBitIdx);
    atomicOr(&destValid[destWordIdx], outBit);
  }
}
