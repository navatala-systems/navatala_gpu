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
__kernel void navatala_graph_compute_modularity_f64(__global const double* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global const double* totalEdgeWeight, __global double* modularity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    double degreeAccum = as_double(0x0000000000000000ul);
    double intraWeightAccum = as_double(0x0000000000000000ul);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double edgeWeight = adjacencyValues[idxU32];
        uint neighbor = colInd[idxU32];
        uint neighborCluster = clusters[neighbor];
        double currentDegree = degreeAccum;
        double newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        double weightToAdd = ((sameCluster) ? (edgeWeight) : (as_double(0x0000000000000000ul)));
        double currentIntraWeight = intraWeightAccum;
        double newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    double degree = degreeAccum;
    double intraWeight = intraWeightAccum;
    double invTotalWeight = (as_double(0x3ff0000000000000ul) / totalWeight);
    double eContrib = (intraWeight * invTotalWeight);
    double degreeNorm = (degree * invTotalWeight);
    double aContrib = (degreeNorm * degreeNorm);
    double contribution = (eContrib - aContrib);
    atomic_add(&modularity[0], contribution);
  }
}
