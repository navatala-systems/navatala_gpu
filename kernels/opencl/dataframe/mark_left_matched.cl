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

__kernel void navatala_dataframe_mark_left_matched(__global const int* leftMatches, __global const uint* probeCount, __global uint* leftMatched) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = probeCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int leftIdx = leftMatches[gid];
    bool hasMatch = (leftIdx >= 0);
    if (hasMatch) {
      uint leftIdxU = ((uint)(leftIdx));
      uint wordIdx = (leftIdxU / (uint)(32u));
      uint bitIdx = (leftIdxU % (uint)(32u));
      uint bitMask = ((uint)(1u) << bitIdx);
      atomic_or(&leftMatched[wordIdx], bitMask);
    }
  }
}
