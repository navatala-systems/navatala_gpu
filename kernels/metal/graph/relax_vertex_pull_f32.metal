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

kernel void navatala_graph_relax_vertex_pull_f32(device const uint* inOffsets [[buffer(0)]], device const uint* inSrcs [[buffer(1)]], device const float* inWeights [[buffer(2)]], device float* dist [[buffer(3)]], device const uint* numVertices [[buffer(4)]], device const float* inf [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if (gid < numV) {
    uint base = inOffsets[gid];
    uint endv = inOffsets[(gid + 1u)];
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
