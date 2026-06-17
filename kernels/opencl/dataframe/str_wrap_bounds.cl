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

__kernel void navatala_dataframe_str_wrap_bounds(__global const int* offsets, __global const char* chars, __global const uint* validity, __global const int* maxWidth, __global const uint* count, __global int* wrapPos) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  int width = maxWidth[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    int startOffset = offsets[gid];
    uint nextIdx = (gid + (uint)(1u));
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
    uint idx0 = ((uint)((startOffset + pos0)));
    uint idx1 = ((uint)((startOffset + pos1)));
    uint idx2 = ((uint)((startOffset + pos2)));
    uint idx3 = ((uint)((startOffset + pos3)));
    char c0 = ((hasPos0) ? (chars[idx0]) : (0));
    char c1 = ((hasPos1) ? (chars[idx1]) : (0));
    char c2 = ((hasPos2) ? (chars[idx2]) : (0));
    char c3 = ((hasPos3) ? (chars[idx3]) : (0));
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
