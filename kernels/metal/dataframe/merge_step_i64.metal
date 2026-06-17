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

kernel void navatala_dataframe_merge_step_i64(device const long* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* blockSize [[buffer(2)]], device long* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint bSize = blockSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint mergeBlockSize = (bSize * 2u);
    uint mergeBlockIdx = (gid / mergeBlockSize);
    uint posInMerge = (gid % mergeBlockSize);
    uint leftStart = (mergeBlockIdx * mergeBlockSize);
    uint rightStart = (leftStart + bSize);
    uint leftEnd = (((rightStart < n)) ? (rightStart) : (n));
    uint rightEnd = ((((leftStart + mergeBlockSize) < n)) ? ((leftStart + mergeBlockSize)) : (n));
    uint leftSize = (leftEnd - leftStart);
    uint rightSize = (rightEnd - rightStart);
    bool leftDone = (posInMerge >= leftSize);
    uint leftIdx = (((posInMerge < leftSize)) ? (posInMerge) : ((leftSize - 1u)));
    uint rightIdx = ((leftDone) ? ((posInMerge - leftSize)) : (0u));
    uint absLeftIdx = (leftStart + leftIdx);
    uint absRightIdx = (rightStart + rightIdx);
    long leftVal = _input[absLeftIdx];
    long rightVal = (((absRightIdx < rightEnd)) ? (_input[absRightIdx]) : (9223372036854775807));
    bool rightExhausted = (absRightIdx >= rightEnd);
    bool leftSmaller = (leftVal <= rightVal);
    bool takeLeft = ((!leftDone) && (rightExhausted || leftSmaller));
    long outVal = ((takeLeft) ? (leftVal) : (rightVal));
    _output[gid] = outVal;
  }
}
