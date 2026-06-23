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

__kernel void navatala_graph_symmetrize_reverse_u32(__global const uint* srcs, __global const uint* dsts, __global const uint* numEdges, __global uint* outSrcs, __global uint* outDsts) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numE = numEdges[0];
  if (gid < numE) {
    uint u = srcs[gid];
    uint v = dsts[gid];
    outSrcs[gid] = v;
    outDsts[gid] = u;
  }
}
