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
extern "C" __global__ void navatala_graph_analyze_partition_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, double* edgeCut, double* cost) {
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
    double nodeEdgeCut = __longlong_as_double(0x0000000000000000ull);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      unsigned int edgeIdxU = ((unsigned int)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        unsigned int neighbor = colInd[edgeIdxU];
        unsigned int neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          double edgeWeight = adjacencyValues[edgeIdxU];
          double currentEdgeCut = nodeEdgeCut;
          double newEdgeCut = (currentEdgeCut + edgeWeight);
          nodeEdgeCut = newEdgeCut;
        }
      }
    }
    double finalNodeEdgeCut = nodeEdgeCut;
    bool hasEdgeCut = (finalNodeEdgeCut > __longlong_as_double(0x0000000000000000ull));
    if (hasEdgeCut) {
      atomicAdd(&edgeCut[0], finalNodeEdgeCut);
    }
    double nValF = ((double)(nVal));
    double nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > __longlong_as_double(0x0000000000000000ull));
    if (hasCostContrib) {
      atomicAdd(&cost[0], nodeCostContrib);
    }
  }
}
