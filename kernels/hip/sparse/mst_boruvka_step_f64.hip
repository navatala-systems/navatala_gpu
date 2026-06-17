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
extern "C" __global__ void navatala_sparse_mst_boruvka_step_f64(const unsigned int* srcNodes, const unsigned int* dstNodes, const double* weights, const unsigned int* numEdges, const unsigned int* numNodes, unsigned int* components, unsigned int* minEdgeIdx, double* minEdgeWeight, bool* mstEdges) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((tid < numEdges[0u])) {
    unsigned int src = srcNodes[tid];
    unsigned int dst = dstNodes[tid];
    unsigned int srcRoot = components[src];
    unsigned int dstRoot = components[dst];
    if ((srcRoot != dstRoot)) {
      double w = weights[tid];
      double currMin = minEdgeWeight[srcRoot];
      if ((w < currMin)) {
        atomicMin(&minEdgeWeight[srcRoot], w);
        minEdgeIdx[srcRoot] = tid;
      }
    }
  }
}
