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

__kernel void navatala_graph_relax_vertex_pull_f32(__global const uint* inOffsets, __global const uint* inSrcs, __global const float* inWeights, __global float* dist, __global const uint* numVertices, __global const float* inf) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = inOffsets[gid];
    uint endv = inOffsets[(gid + (uint)(1u))];
    uint deg = (endv - base);
    float infV = inf[0];
    float best = dist[gid];
    for (int k = 0; k < (int)(deg); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint u = inSrcs[eidx];
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
