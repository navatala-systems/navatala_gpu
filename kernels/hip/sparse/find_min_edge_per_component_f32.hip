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
extern "C" __global__ void navatala_sparse_find_min_edge_per_component_f32(const unsigned int* srcNodes, const unsigned int* dstNodes, const float* weights, const unsigned int* components, const unsigned int* numEdges, const unsigned int* numNodes, unsigned int* minEdgeIdx, float* minEdgeWeight) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int edgeIdx = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (edgeIdx < numEdges[0u]) {
    unsigned int src = srcNodes[edgeIdx];
    unsigned int dst = dstNodes[edgeIdx];
    unsigned int srcComp = components[src];
    unsigned int dstComp = components[dst];
    if (srcComp != dstComp) {
      float w = weights[edgeIdx];
      float currWeight = minEdgeWeight[srcComp];
      if (w < currWeight) {
        atomicMin(&minEdgeWeight[srcComp], w);
        minEdgeIdx[srcComp] = edgeIdx;
      }
    }
  }
}
