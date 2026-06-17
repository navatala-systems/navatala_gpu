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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_graph_pair_intersection_u32(const unsigned int* offsets, const unsigned int* indices, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, unsigned int* inter) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if ((gid < numP)) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int baseU = offsets[a];
    unsigned int endU = offsets[(a + 1u)];
    unsigned int baseV = offsets[b];
    unsigned int endV = offsets[(b + 1u)];
    unsigned int degU = (endU - baseU);
    unsigned int interAccum = 0u;
    unsigned int loAccum = 0u;
    unsigned int hiAccum = 0u;
    for (int k = 0; k < (int)(degU); ++k) {
      unsigned int eidx = (baseU + ((unsigned int)(k)));
      unsigned int w = indices[eidx];
      loAccum = baseV;
      hiAccum = endV;
      for (int _bs = 0; _bs < (int)(32u); ++_bs) {
        unsigned int lo = loAccum;
        unsigned int hi = hiAccum;
        if ((lo < hi)) {
          unsigned int mid = ((lo + hi) / 2u);
          unsigned int midval = indices[mid];
          bool goRight = (midval < w);
          unsigned int newLo = ((goRight) ? ((mid + 1u)) : (lo));
          unsigned int newHi = ((goRight) ? (hi) : (mid));
          loAccum = newLo;
          hiAccum = newHi;
        }
      }
      unsigned int lb = loAccum;
      if ((lb < endV)) {
        unsigned int cand = indices[lb];
        if ((cand == w)) {
          interAccum = (interAccum + 1u);
        }
      }
    }
    inter[gid] = interAccum;
  }
}
