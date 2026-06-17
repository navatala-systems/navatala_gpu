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

__kernel void navatala_dataframe_str_contains_char(__global const int* offsets, __global const char* chars, __global const uint* validity, __global const char* targetChar, __global const uint* count, __global uint* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  char target = targetChar[(uint)(0u)];
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
    uint startU = ((uint)(startOffset));
    uint endU = ((uint)(endOffset));
    uint len = (endU - startU);
    bool hasLen1 = (len >= (uint)(1u));
    uint idx0 = startU;
    char c0 = chars[idx0];
    bool match0 = (hasLen1 && (c0 == target));
    bool hasLen2 = (len >= (uint)(2u));
    uint idx1 = (startU + (uint)(1u));
    char c1 = chars[idx1];
    bool match1 = (hasLen2 && (c1 == target));
    bool hasLen3 = (len >= (uint)(3u));
    uint idx2 = (startU + (uint)(2u));
    char c2 = chars[idx2];
    bool match2 = (hasLen3 && (c2 == target));
    bool hasLen4 = (len >= (uint)(4u));
    uint idx3 = (startU + (uint)(3u));
    char c3 = chars[idx3];
    bool match3 = (hasLen4 && (c3 == target));
    bool anyMatch = (match0 || (match1 || (match2 || match3)));
    uint result = (((isValid && anyMatch)) ? ((uint)(1u)) : ((uint)(0u)));
    _output[gid] = result;
    uint outBit = (validBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
