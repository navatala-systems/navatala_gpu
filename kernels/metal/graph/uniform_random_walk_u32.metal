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

kernel void navatala_graph_uniform_random_walk_u32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* startVertices [[buffer(2)]], device const uint* maxLen [[buffer(3)]], device const uint* numWalks [[buffer(4)]], device const uint* seed [[buffer(5)]], device uint* walks [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numW = numWalks[0];
  if ((gid < numW)) {
    uint maxL = maxLen[0];
    uint seedV = seed[0];
    uint rowBase = (gid * maxL);
    uint curAccum = startVertices[gid];
    uint rngAccum = (seedV + gid);
    for (int s = 0; s < (int)(maxL); ++s) {
      uint cur = curAccum;
      uint pos = (rowBase + ((uint)(s)));
      walks[pos] = cur;
      uint off = offsets[cur];
      uint off1 = offsets[(cur + 1u)];
      uint deg = (off1 - off);
      uint rng = rngAccum;
      uint rngNext = ((rng * 1103515245u) + 12345u);
      rngAccum = rngNext;
      if ((deg > 0u)) {
        uint pick = (rngNext % deg);
        uint nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}
