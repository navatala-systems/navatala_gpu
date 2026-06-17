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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_str_wrap_bounds(device const int* offsets [[buffer(0)]], device const char* chars [[buffer(1)]], device const uint* validity [[buffer(2)]], device const int* maxWidth [[buffer(3)]], device const uint* count [[buffer(4)]], device int* wrapPos [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  int width = maxWidth[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int startOffset = offsets[gid];
    uint nextIdx = (gid + 1u);
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
