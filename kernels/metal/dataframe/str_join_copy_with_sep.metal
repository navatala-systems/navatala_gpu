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

kernel void navatala_dataframe_str_join_copy_with_sep(device const char* chars [[buffer(0)]], device const int* offsets [[buffer(1)]], device const uint* validity [[buffer(2)]], device const char* sepChar [[buffer(3)]], device const uint* count [[buffer(4)]], device const int* outputPositions [[buffer(5)]], device char* outputChars [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  char separator = sepChar[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int srcStart = offsets[gid];
    uint nextIdx = (gid + 1u);
    int srcEnd = offsets[nextIdx];
    int strLen = (srcEnd - srcStart);
    if (isValid) {
      int outPos = outputPositions[gid];
      bool needSep = (outPos > 0);
      int sepPos = (outPos - 1);
      uint sepIdx = ((uint)(sepPos));
      if (needSep) {
        outputChars[sepIdx] = separator;
      }
      uint charStartIdx = ((uint)(outPos));
      uint srcStartU = ((uint)(srcStart));
      bool hasChar0 = (strLen > 0);
      uint srcIdx0 = srcStartU;
      char char0 = chars[srcIdx0];
      uint dstIdx0 = charStartIdx;
      if (hasChar0) {
        outputChars[dstIdx0] = char0;
      }
      bool hasChar1 = (strLen > 1);
      uint srcIdx1 = (srcStartU + 1u);
      char char1 = chars[srcIdx1];
      uint dstIdx1 = (charStartIdx + 1u);
      if (hasChar1) {
        outputChars[dstIdx1] = char1;
      }
      bool hasChar2 = (strLen > 2);
      uint srcIdx2 = (srcStartU + 2u);
      char char2 = chars[srcIdx2];
      uint dstIdx2 = (charStartIdx + 2u);
      if (hasChar2) {
        outputChars[dstIdx2] = char2;
      }
      bool hasChar3 = (strLen > 3);
      uint srcIdx3 = (srcStartU + 3u);
      char char3 = chars[srcIdx3];
      uint dstIdx3 = (charStartIdx + 3u);
      if (hasChar3) {
        outputChars[dstIdx3] = char3;
      }
    }
  }
}
