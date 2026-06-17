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

kernel void navatala_dataframe_probe_hash_table_i32(device const int* rightKeys [[buffer(0)]], device const uint* rightValidity [[buffer(1)]], device const uint* rightHashes [[buffer(2)]], device const uint* rightCount [[buffer(3)]], device const int* tableKeys [[buffer(4)]], device const uint* tableRowIds [[buffer(5)]], device const uint* tableSize [[buffer(6)]], device int* leftMatches [[buffer(7)]], device uint* rightMatches [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = rightCount[0u];
  uint tSize = tableSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = rightValidity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int rightKey = rightKeys[gid];
    uint startIdx = rightHashes[gid];
    uint probeIdx = startIdx;
    bool foundMatch = false;
    bool done = (!isValid);
    uint matchedRowId = 0u;
    for (int __iter = 0; __iter < 65536; ++__iter) {
      if (!((!done))) break;
      uint curIdx = probeIdx;
      int tableKey = tableKeys[curIdx];
      bool isEmpty = (tableKey == 2147483647);
      bool keyMatch = (rightKey == tableKey);
      if (isEmpty) {
        done = true;
      } else {
        if (keyMatch) {
          uint rowId = tableRowIds[curIdx];
          matchedRowId = rowId;
          foundMatch = true;
          done = true;
        } else {
          uint nextIdx = ((curIdx + 1u) % tSize);
          probeIdx = nextIdx;
        }
      }
    }
    int leftMatch = ((foundMatch) ? (((int)(matchedRowId))) : (-1));
    leftMatches[gid] = leftMatch;
    rightMatches[gid] = gid;
  }
}
