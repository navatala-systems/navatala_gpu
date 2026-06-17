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

kernel void navatala_dataframe_cross_join_gather_i32(device const int* leftData [[buffer(0)]], device const int* rightData [[buffer(1)]], device const uint* leftCount [[buffer(2)]], device const uint* rightCount [[buffer(3)]], device const ulong* outputCount [[buffer(4)]], device int* leftOutput [[buffer(5)]], device int* rightOutput [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  ulong gid = ((ulong)(int(__gid.x)));
  ulong n = outputCount[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint rCount = rightCount[0u];
    ulong rCount64 = ((ulong)(rCount));
    ulong leftIdx64 = (gid / rCount64);
    ulong rightIdx64 = (gid % rCount64);
    uint leftIdx = ((uint)(leftIdx64));
    uint rightIdx = ((uint)(rightIdx64));
    int leftVal = leftData[leftIdx];
    int rightVal = rightData[rightIdx];
    uint outIdx = ((uint)(gid));
    leftOutput[outIdx] = leftVal;
    rightOutput[outIdx] = rightVal;
  }
}
