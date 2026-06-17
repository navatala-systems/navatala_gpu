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

kernel void navatala_dataframe_hash_column_row(device const uint* colHashes1 [[buffer(0)]], device const uint* colHashes2 [[buffer(1)]], device const uint* count [[buffer(2)]], device uint* rowHashes [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint h1 = colHashes1[gid];
    uint h2 = colHashes2[gid];
    uint c1 = 3432918353u;
    uint c2 = 461845907u;
    uint h1Left = (h1 << 13u);
    uint h1Right = (h1 >> 19u);
    uint h1Rotated = (h1Left | h1Right);
    uint combined1 = (h1Rotated ^ h2);
    uint mixed1 = (combined1 * c1);
    uint m1Left = (mixed1 << 15u);
    uint m1Right = (mixed1 >> 17u);
    uint m1Rotated = (m1Left | m1Right);
    uint mixed2 = (m1Rotated * c2);
    uint finalMix = (mixed2 ^ (mixed2 >> 16u));
    rowHashes[gid] = finalMix;
  }
}
