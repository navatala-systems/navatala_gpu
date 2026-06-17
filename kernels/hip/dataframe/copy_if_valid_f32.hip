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
extern "C" __global__ void navatala_dataframe_copy_if_valid_f32(const float* src, const unsigned int* srcValid, const unsigned int* dstValid, const unsigned int* count, float* dst, unsigned int* outValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int srcValidWord = srcValid[wordIdx];
    unsigned int srcValidBit = ((srcValidWord >> bitIdx) & 1u);
    bool srcIsValid = (srcValidBit == 1u);
    unsigned int dstValidWord = dstValid[wordIdx];
    unsigned int dstValidBit = ((dstValidWord >> bitIdx) & 1u);
    bool dstIsValid = (dstValidBit == 1u);
    bool bothValid = (srcIsValid && dstIsValid);
    if (bothValid) {
      float val = src[gid];
      dst[gid] = val;
      unsigned int outBit = (1u << bitIdx);
      atomicOr(&outValid[wordIdx], outBit);
    }
  }
}
