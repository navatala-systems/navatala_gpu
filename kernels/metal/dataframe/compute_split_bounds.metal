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

kernel void navatala_dataframe_compute_split_bounds(device const uint* totalRows [[buffer(0)]], device const uint* numSplits [[buffer(1)]], device int* splitStarts [[buffer(2)]], device int* splitEnds [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nRows = totalRows[0u];
  uint nSplits = numSplits[0u];
  bool inBounds = (gid < nSplits);
  if (inBounds) {
    uint baseSize = (nRows / nSplits);
    uint remainder = (nRows % nSplits);
    bool hasExtra = (gid < remainder);
    uint extraRows = ((hasExtra) ? (1u) : (0u));
    uint splitSize = (baseSize + extraRows);
    uint prevSplitsWithExtra = (((gid < remainder)) ? (gid) : (remainder));
    uint prevSplitsWithoutExtra = (gid - prevSplitsWithExtra);
    uint sizePlusOne = (baseSize + 1u);
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
