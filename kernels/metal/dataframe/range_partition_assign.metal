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

kernel void navatala_dataframe_range_partition_assign(device const int* keys [[buffer(0)]], device const int* bounds [[buffer(1)]], device const uint* numPartitions [[buffer(2)]], device const uint* count [[buffer(3)]], device uint* partitionIds [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint numParts = numPartitions[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    uint numBounds = (numParts - 1u);
    int b0 = (((numBounds >= 1u)) ? (bounds[0u]) : (2147483647));
    int b1 = (((numBounds >= 2u)) ? (bounds[1u]) : (2147483647));
    int b2 = (((numBounds >= 3u)) ? (bounds[2u]) : (2147483647));
    int b3 = (((numBounds >= 4u)) ? (bounds[3u]) : (2147483647));
    uint ge0 = (((key >= b0)) ? (1u) : (0u));
    uint ge1 = (((key >= b1)) ? (1u) : (0u));
    uint ge2 = (((key >= b2)) ? (1u) : (0u));
    uint ge3 = (((key >= b3)) ? (1u) : (0u));
    uint partId = (ge0 + (ge1 + (ge2 + ge3)));
    uint maxPartId = (numParts - 1u);
    uint clampedPartId = (((partId < maxPartId)) ? (partId) : (maxPartId));
    partitionIds[gid] = clampedPartId;
  }
}
