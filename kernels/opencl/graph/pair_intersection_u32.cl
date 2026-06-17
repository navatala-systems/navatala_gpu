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

__kernel void navatala_graph_pair_intersection_u32(__global const uint* offsets, __global const uint* indices, __global const uint* pairsA, __global const uint* pairsB, __global const uint* numPairs, __global uint* inter) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numP = numPairs[0];
  if ((gid < numP)) {
    uint a = pairsA[gid];
    uint b = pairsB[gid];
    uint baseU = offsets[a];
    uint endU = offsets[(a + (uint)(1u))];
    uint baseV = offsets[b];
    uint endV = offsets[(b + (uint)(1u))];
    uint degU = (endU - baseU);
    uint interAccum = (uint)(0u);
    uint loAccum = (uint)(0u);
    uint hiAccum = (uint)(0u);
    for (int k = 0; k < (int)(degU); ++k) {
      uint eidx = (baseU + ((uint)(k)));
      uint w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)((uint)(32u)); ++_bs) {
        uint lo = loAccum;
        uint hi = hiAccum;
        if ((lo < hi)) {
          uint mid = ((lo + hi) / (uint)(2u));
          uint midval = indices[mid];
          bool goRight = (midval < w);
          uint newLo = ((goRight) ? ((mid + (uint)(1u))) : (lo));
          uint newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      uint lb = loAccum;
      if ((lb < endV)) {
        uint cand = indices[lb];
        if ((cand == w)) {
          interAccum = (interAccum + (uint)(1u));
        }
      }
    }
    inter[gid] = interAccum;
  }
}
