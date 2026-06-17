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
extern "C" __global__ void navatala_dataframe_str_join_copy_with_sep(const signed char* chars, const int* offsets, const unsigned int* validity, const signed char* sepChar, const unsigned int* count, const int* outputPositions, signed char* outputChars) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  signed char separator = sepChar[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int srcStart = offsets[gid];
    unsigned int nextIdx = (gid + 1u);
    int srcEnd = offsets[nextIdx];
    int strLen = (srcEnd - srcStart);
    if (isValid) {
      int outPos = outputPositions[gid];
      bool needSep = (outPos > 0);
      int sepPos = (outPos - 1);
      unsigned int sepIdx = ((unsigned int)(sepPos));
      if (needSep) {
        outputChars[sepIdx] = separator;
      }
      unsigned int charStartIdx = ((unsigned int)(outPos));
      unsigned int srcStartU = ((unsigned int)(srcStart));
      bool hasChar0 = (strLen > 0);
      unsigned int srcIdx0 = srcStartU;
      signed char char0 = chars[srcIdx0];
      unsigned int dstIdx0 = charStartIdx;
      if (hasChar0) {
        outputChars[dstIdx0] = char0;
      }
      bool hasChar1 = (strLen > 1);
      unsigned int srcIdx1 = (srcStartU + 1u);
      signed char char1 = chars[srcIdx1];
      unsigned int dstIdx1 = (charStartIdx + 1u);
      if (hasChar1) {
        outputChars[dstIdx1] = char1;
      }
      bool hasChar2 = (strLen > 2);
      unsigned int srcIdx2 = (srcStartU + 2u);
      signed char char2 = chars[srcIdx2];
      unsigned int dstIdx2 = (charStartIdx + 2u);
      if (hasChar2) {
        outputChars[dstIdx2] = char2;
      }
      bool hasChar3 = (strLen > 3);
      unsigned int srcIdx3 = (srcStartU + 3u);
      signed char char3 = chars[srcIdx3];
      unsigned int dstIdx3 = (charStartIdx + 3u);
      if (hasChar3) {
        outputChars[dstIdx3] = char3;
      }
    }
  }
}
