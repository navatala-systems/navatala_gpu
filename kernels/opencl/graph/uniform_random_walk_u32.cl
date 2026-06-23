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

__kernel void navatala_graph_uniform_random_walk_u32(__global const uint* offsets, __global const uint* indices, __global const uint* startVertices, __global const uint* maxLen, __global const uint* numWalks, __global const uint* seed, __global uint* walks) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numW = numWalks[0];
  if (gid < numW) {
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
      uint off1 = offsets[(cur + (uint)(1u))];
      uint deg = (off1 - off);
      uint rng = rngAccum;
      uint rngNext = ((rng * (uint)(1103515245u)) + (uint)(12345u));
      rngAccum = rngNext;
      if (deg > (uint)(0u)) {
        uint pick = (rngNext % deg);
        uint nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}
