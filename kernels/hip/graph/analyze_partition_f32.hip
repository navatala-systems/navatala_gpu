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
extern "C" __global__ void navatala_graph_analyze_partition_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, float* edgeCut, float* cost) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float nodeEdgeCut = __uint_as_float(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          float edgeWeight = adjacencyValues[edgeIdxU];
          float currentEdgeCut = nodeEdgeCut;
          float newEdgeCut = (currentEdgeCut + edgeWeight);
          nodeEdgeCut = newEdgeCut;
        }
      }
    }
    float finalNodeEdgeCut = nodeEdgeCut;
    bool hasEdgeCut = (finalNodeEdgeCut > __uint_as_float(0x00000000u));
    if (hasEdgeCut) {
      atomicAdd(&edgeCut[0], finalNodeEdgeCut);
    }
    float nValF = ((float)(nVal));
    float nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > __uint_as_float(0x00000000u));
    if (hasCostContrib) {
      atomicAdd(&cost[0], nodeCostContrib);
    }
  }
}
