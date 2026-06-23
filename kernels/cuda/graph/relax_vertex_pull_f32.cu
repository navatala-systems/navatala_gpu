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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_graph_relax_vertex_pull_f32(const unsigned int* inOffsets, const unsigned int* inSrcs, const float* inWeights, float* dist, const unsigned int* numVertices, const float* inf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numV = numVertices[0];
  if (gid < numV) {
    unsigned int base = inOffsets[gid];
    unsigned int endv = inOffsets[(gid + 1u)];
    unsigned int deg = (endv - base);
    float infV = inf[0];
    float best = dist[gid];
    for (int k = 0; k < (int)(deg); ++k) {
      unsigned int eidx = (base + ((unsigned int)(k)));
      unsigned int u = inSrcs[eidx];
      float w = inWeights[eidx];
      float du = dist[u];
      if (du < infV) {
        float cand = (du + w);
        float cur = best;
        best = (((cand < cur)) ? (cand) : (cur));
      }
    }
    dist[gid] = best;
  }
}
