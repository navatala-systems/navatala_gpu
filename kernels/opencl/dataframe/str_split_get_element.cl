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

__kernel void navatala_dataframe_str_split_get_element(__global const int* offsets, __global const char* chars, __global const uint* validity, __global const char* delimChar, __global const int* elementIdx, __global const uint* count, __global int* elementStart, __global int* elementEnd, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  char delim = delimChar[(uint)(0u)];
  int elemIdx = elementIdx[(uint)(0u)];
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
    bool wantsFirst = (elemIdx == 0);
    uint idx0 = ((uint)(startOffset));
    uint idx1 = (idx0 + (uint)(1u));
    uint idx2 = (idx0 + (uint)(2u));
    uint idx3 = (idx0 + (uint)(3u));
    char c0 = (((length >= 1)) ? (chars[idx0]) : (0));
    char c1 = (((length >= 2)) ? (chars[idx1]) : (0));
    char c2 = (((length >= 3)) ? (chars[idx2]) : (0));
    char c3 = (((length >= 4)) ? (chars[idx3]) : (0));
    bool isDelim0 = (c0 == delim);
    bool isDelim1 = (c1 == delim);
    bool isDelim2 = (c2 == delim);
    bool isDelim3 = (c3 == delim);
    int firstDelimPos = ((isDelim0) ? (0) : (((isDelim1) ? (1) : (((isDelim2) ? (2) : (((isDelim3) ? (3) : (-1))))))));
    bool hasDelim = (firstDelimPos != -1);
    int elem0Start = startOffset;
    int elem0End = ((hasDelim) ? ((startOffset + firstDelimPos)) : (endOffset));
    int elem1Start = ((hasDelim) ? ((elem0End + 1)) : (-1));
    int elem1End = ((hasDelim) ? (endOffset) : (-1));
    int elemStart = ((wantsFirst) ? (elem0Start) : (elem1Start));
    int elemEnd = ((wantsFirst) ? (elem0End) : (elem1End));
    int outStart = ((isValid) ? (elemStart) : (-1));
    int outEnd = ((isValid) ? (elemEnd) : (-1));
    elementStart[gid] = outStart;
    elementEnd[gid] = outEnd;
    uint outBit = (validBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
