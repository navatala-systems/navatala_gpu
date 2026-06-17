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

kernel void navatala_dataframe_hash_partition_i32(device const int* keys [[buffer(0)]], device const uint* numPartitions [[buffer(1)]], device const uint* count [[buffer(2)]], device uint* partitionIds [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint numParts = numPartitions[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    int key = keys[gid];
    uint k = ((uint)(key));
    uint h1 = (k ^ (k >> 16u));
    uint h2 = (h1 * 2246822507u);
    uint h3 = (h2 ^ (h2 >> 13u));
    uint h4 = (h3 * 3266489909u);
    uint hash = (h4 ^ (h4 >> 16u));
    uint partId = (hash % numParts);
    partitionIds[gid] = partId;
  }
}
