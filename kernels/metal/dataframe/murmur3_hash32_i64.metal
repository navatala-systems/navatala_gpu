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

kernel void navatala_dataframe_murmur3_hash32_i64(device const long* keys [[buffer(0)]], device const uint* seed [[buffer(1)]], device const uint* count [[buffer(2)]], device uint* hashes [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint s = seed[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    long key = keys[gid];
    uint kLow = ((uint)(key));
    uint kHigh = ((uint)((key >> 32)));
    uint c1 = 3432918353u;
    uint c2 = 461845907u;
    uint k1a = (kLow * c1);
    uint k1b = ((k1a << 15u) | (k1a >> 17u));
    uint k1c = (k1b * c2);
    uint h1a = (s ^ k1c);
    uint h1b = ((h1a << 13u) | (h1a >> 19u));
    uint h1c = ((h1b * 5u) + 3864292196u);
    uint k2a = (kHigh * c1);
    uint k2b = ((k2a << 15u) | (k2a >> 17u));
    uint k2c = (k2b * c2);
    uint h2a = (h1c ^ k2c);
    uint h2b = ((h2a << 13u) | (h2a >> 19u));
    uint h2c = ((h2b * 5u) + 3864292196u);
    uint h3 = (h2c ^ 8u);
    uint h4 = (h3 ^ (h3 >> 16u));
    uint h5 = (h4 * 2246822507u);
    uint h6 = (h5 ^ (h5 >> 13u));
    uint h7 = (h6 * 3266489909u);
    uint hash = (h7 ^ (h7 >> 16u));
    hashes[gid] = hash;
  }
}
