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

__kernel void navatala_dataframe_count_unmatched_left(__global const uint* leftMatched, __global const uint* leftRowCount, __global uint* unmatchedCount) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = leftRowCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint word = leftMatched[wordIdx];
    uint bit = ((word >> bitIdx) & (uint)(1u));
    bool isMatched = (bit == (uint)(1u));
    if ((isMatched == false)) {
      atomic_add(&unmatchedCount[(uint)(0u)], (uint)(1u));
    }
  }
}
