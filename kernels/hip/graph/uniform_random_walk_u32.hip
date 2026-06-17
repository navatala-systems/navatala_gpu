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
extern "C" __global__ void navatala_graph_uniform_random_walk_u32(const unsigned int* offsets, const unsigned int* indices, const unsigned int* startVertices, const unsigned int* maxLen, const unsigned int* numWalks, const unsigned int* seed, unsigned int* walks) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numW = numWalks[0];
  if ((gid < numW)) {
    unsigned int maxL = maxLen[0];
    unsigned int seedV = seed[0];
    unsigned int rowBase = (gid * maxL);
    unsigned int curAccum = startVertices[gid];
    unsigned int rngAccum = (seedV + gid);
    for (int s = 0; s < (int)(maxL); ++s) {
      unsigned int cur = curAccum;
      unsigned int pos = (rowBase + ((unsigned int)(s)));
      walks[pos] = cur;
      unsigned int off = offsets[cur];
      unsigned int off1 = offsets[(cur + 1u)];
      unsigned int deg = (off1 - off);
      unsigned int rng = rngAccum;
      unsigned int rngNext = ((rng * 1103515245u) + 12345u);
      rngAccum = rngNext;
      if ((deg > 0u)) {
        unsigned int pick = (rngNext % deg);
        unsigned int nbr = indices[(off + pick)];
        curAccum = nbr;
      }
    }
  }
}
