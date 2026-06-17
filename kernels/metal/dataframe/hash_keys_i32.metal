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

kernel void navatala_dataframe_hash_keys_i32(device const int* keys [[buffer(0)]], device const uint* validity [[buffer(1)]], device const uint* count [[buffer(2)]], device const uint* tableSize [[buffer(3)]], device uint* hashes [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint tSize = tableSize[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int key = keys[gid];
    uint k1 = ((uint)(key));
    uint k2 = (k1 * 3432918353u);
    uint k3 = ((k2 << 15u) | (k2 >> 17u));
    uint k4 = (k3 * 461845907u);
    uint h1 = (k4 ^ 2246822507u);
    uint h2 = (h1 ^ (h1 >> 16u));
    uint h3 = (h2 * 3266489909u);
    uint h4 = (h3 ^ (h3 >> 13u));
    uint hashIdx = (h4 % tSize);
    uint outHash = ((isValid) ? (hashIdx) : (2147483647u));
    hashes[gid] = outHash;
  }
}
