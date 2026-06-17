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
extern "C" __global__ void navatala_dataframe_str_wrap_bounds(const int* offsets, const signed char* chars, const unsigned int* validity, const int* maxWidth, const unsigned int* count, int* wrapPos) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  int width = maxWidth[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int startOffset = offsets[gid];
    unsigned int nextIdx = (gid + 1u);
    int endOffset = offsets[nextIdx];
    int length = (endOffset - startOffset);
    bool needsWrap = (length > width);
    int widthBound = (((width < length)) ? (width) : (length));
    int pos0 = (widthBound - 1);
    int pos1 = (widthBound - 2);
    int pos2 = (widthBound - 3);
    int pos3 = (widthBound - 4);
    bool hasPos0 = (pos0 >= 0);
    bool hasPos1 = (pos1 >= 0);
    bool hasPos2 = (pos2 >= 0);
    bool hasPos3 = (pos3 >= 0);
    unsigned int idx0 = ((unsigned int)((startOffset + pos0)));
    unsigned int idx1 = ((unsigned int)((startOffset + pos1)));
    unsigned int idx2 = ((unsigned int)((startOffset + pos2)));
    unsigned int idx3 = ((unsigned int)((startOffset + pos3)));
    signed char c0 = ((hasPos0) ? (chars[idx0]) : (0));
    signed char c1 = ((hasPos1) ? (chars[idx1]) : (0));
    signed char c2 = ((hasPos2) ? (chars[idx2]) : (0));
    signed char c3 = ((hasPos3) ? (chars[idx3]) : (0));
    bool isSpace0 = (((int)(c0)) == 32);
    bool isSpace1 = (((int)(c1)) == 32);
    bool isSpace2 = (((int)(c2)) == 32);
    bool isSpace3 = (((int)(c3)) == 32);
    bool spaceAt0 = (hasPos0 && isSpace0);
    bool spaceAt1 = (hasPos1 && isSpace1);
    bool spaceAt2 = (hasPos2 && isSpace2);
    bool spaceAt3 = (hasPos3 && isSpace3);
    int wrapPosition = ((spaceAt0) ? ((pos0 + 1)) : (((spaceAt1) ? ((pos1 + 1)) : (((spaceAt2) ? ((pos2 + 1)) : (((spaceAt3) ? ((pos3 + 1)) : (widthBound))))))));
    int result = (((isValid && needsWrap)) ? (wrapPosition) : (-1));
    wrapPos[gid] = result;
  }
}
