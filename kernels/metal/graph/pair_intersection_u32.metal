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

kernel void navatala_graph_pair_intersection_u32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const uint* pairsA [[buffer(2)]], device const uint* pairsB [[buffer(3)]], device const uint* numPairs [[buffer(4)]], device uint* inter [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numP = numPairs[0];
  if (gid < numP) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint baseU = offsets[a];
    uint endU = offsets[(a + 1u)];
    uint baseV = offsets[b];
    uint endV = offsets[(b + 1u)];
    uint degU = (endU - baseU);
    uint interAccum = 0u;
    uint loAccum = 0u;
    uint hiAccum = 0u;
    for (int k = 0; k < (int)(degU); ++k) {
      uint eidx = (baseU + ((uint)(k)));
      uint w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)(32u); ++_bs) {
        uint lo = loAccum;
        uint hi = hiAccum;
        if (lo < hi) {
          uint mid = ((lo + hi) / 2u);
          uint midval = indices[mid];
          bool goRight = (midval < w);
          uint newLo = ((goRight) ? ((mid + 1u)) : (lo));
          uint newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      uint lb = loAccum;
      if (lb < endV) {
        uint cand = indices[lb];
        if (cand == w) {
          interAccum = (interAccum + 1u);
        }
      }
    }
    inter[gid] = interAccum;
  }
}
