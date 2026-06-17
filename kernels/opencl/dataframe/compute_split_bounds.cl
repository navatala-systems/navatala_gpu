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

__kernel void navatala_dataframe_compute_split_bounds(__global const uint* totalRows, __global const uint* numSplits, __global int* splitStarts, __global int* splitEnds) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nRows = totalRows[(uint)(0u)];
  uint nSplits = numSplits[(uint)(0u)];
  bool inBounds = (gid < nSplits);
  if (inBounds) {
    uint baseSize = (nRows / nSplits);
    uint remainder = (nRows % nSplits);
    bool hasExtra = (gid < remainder);
    uint extraRows = ((hasExtra) ? ((uint)(1u)) : ((uint)(0u)));
    uint splitSize = (baseSize + extraRows);
    uint prevSplitsWithExtra = (((gid < remainder)) ? (gid) : (remainder));
    uint prevSplitsWithoutExtra = (gid - prevSplitsWithExtra);
    uint sizePlusOne = (baseSize + (uint)(1u));
    uint startFromExtra = (prevSplitsWithExtra * sizePlusOne);
    uint startFromRegular = (prevSplitsWithoutExtra * baseSize);
    uint startIdx = (startFromExtra + startFromRegular);
    uint endIdx = (startIdx + splitSize);
    int startInt = ((int)(startIdx));
    int endInt = ((int)(endIdx));
    splitStarts[gid] = startInt;
    splitEnds[gid] = endInt;
  }
}
