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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_analyze_partition_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global double* edgeCut, __global double* cost) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint numClustersVal = numClusters[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double nodeEdgeCut = as_double(0x0000000000000000ul);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
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
    bool hasEdgeCut = (finalNodeEdgeCut > as_double(0x0000000000000000ul));
    if (hasEdgeCut) {
      atomic_add(&edgeCut[0], finalNodeEdgeCut);
    }
    double nValF = ((double)(nVal));
    double nodeCostContrib = (finalNodeEdgeCut / nValF);
    bool hasCostContrib = (nodeCostContrib > as_double(0x0000000000000000ul));
    if (hasCostContrib) {
      atomic_add(&cost[0], nodeCostContrib);
    }
  }
}
