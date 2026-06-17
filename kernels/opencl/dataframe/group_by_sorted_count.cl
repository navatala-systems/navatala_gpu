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

__kernel void navatala_dataframe_group_by_sorted_count(__global const int* sortedKeys, __global const uint* validity, __global const uint* count, __global uint* groupCounts, __global int* groupBounds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    int currentKey = sortedKeys[gid];
    bool isFirst = (gid == (uint)(0u));
    uint prevIdx = (gid - (uint)(1u));
    uint safePrevIdx = ((isFirst) ? (gid) : (prevIdx));
    int prevKey = sortedKeys[safePrevIdx];
    bool keyChanged = (currentKey != prevKey);
    bool isBoundary = (isFirst || keyChanged);
    uint groupIdx = ((uint)(currentKey));
    if (isValid) {
      atomic_add(&groupCounts[groupIdx], (uint)(1u));
    }
    if (isBoundary) {
      int boundaryIdx = ((int)(gid));
      groupBounds[groupIdx] = boundaryIdx;
    }
  }
}
