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

__kernel void navatala_dataframe_probe_hash_table_i32(__global const int* rightKeys, __global const uint* rightValidity, __global const uint* rightHashes, __global const uint* rightCount, __global const int* tableKeys, __global const uint* tableRowIds, __global const uint* tableSize, __global int* leftMatches, __global uint* rightMatches) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = rightCount[(uint)(0u)];
  uint tSize = tableSize[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = rightValidity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    int rightKey = rightKeys[gid];
    uint startIdx = rightHashes[gid];
    uint probeIdx = startIdx;
    bool foundMatch = false;
    bool done = (!isValid);
    uint matchedRowId = (uint)(0u);
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
          uint nextIdx = ((curIdx + (uint)(1u)) % tSize);
          probeIdx = nextIdx;
        }
      }
    }
    int leftMatch = ((foundMatch) ? (((int)(matchedRowId))) : (-1));
    leftMatches[gid] = leftMatch;
    rightMatches[gid] = gid;
  }
}
