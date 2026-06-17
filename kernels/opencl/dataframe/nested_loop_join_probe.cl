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

__kernel void navatala_dataframe_nested_loop_join_probe(__global const int* leftKeys, __global const int* rightKeys, __global const uint* leftCount, __global const uint* rightCount, __global const uint* maxMatches, __global uint* matchCount, __global int* matchLeft, __global int* matchRight) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lCount = leftCount[(uint)(0u)];
  uint rCount = rightCount[(uint)(0u)];
  uint maxM = maxMatches[(uint)(0u)];
  bool inBounds = (gid < lCount);
  if (inBounds) {
    int leftKey = leftKeys[gid];
    for (int rightIdx = 0; rightIdx < (int)(rCount); ++rightIdx) {
      int rightKey = rightKeys[rightIdx];
      bool isMatch = (leftKey == rightKey);
      if (isMatch) {
        atomic_add(&matchCount[(uint)(0u)], (uint)(1u));
        uint outPos = matchCount[(uint)(0u)];
        uint outPosAdj = (outPos - (uint)(1u));
        bool canStore = (outPosAdj < maxM);
        if (canStore) {
          int leftIdxI32 = ((int)(gid));
          int rightIdxI32 = ((int)(rightIdx));
          matchLeft[outPosAdj] = leftIdxI32;
          matchRight[outPosAdj] = rightIdxI32;
        }
      }
    }
  }
}
