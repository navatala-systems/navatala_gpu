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
extern "C" __global__ void navatala_dataframe_compact_by_mask_i32(const int* _input, const unsigned int* inputValid, const unsigned int* mask, const unsigned int* prefixSum, const unsigned int* count, int* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int selected = mask[gid];
    bool isSelected = (selected == 1u);
    if (isSelected) {
      unsigned int outIdx = prefixSum[gid];
      int val = _input[gid];
      _output[outIdx] = val;
      unsigned int srcWordIdx = (gid / 32u);
      unsigned int srcBitIdx = (gid % 32u);
      unsigned int srcValidWord = inputValid[srcWordIdx];
      unsigned int srcValidBit = ((srcValidWord >> srcBitIdx) & 1u);
      unsigned int dstWordIdx = (outIdx / 32u);
      unsigned int dstBitIdx = (outIdx % 32u);
      unsigned int dstBit = (srcValidBit << dstBitIdx);
      atomicOr(&outputValid[dstWordIdx], dstBit);
    }
  }
}
