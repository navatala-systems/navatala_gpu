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

__kernel void navatala_graph_compute_modularity_f32(__global const float* adjacencyValues, __global const uint* rowPtr, __global const uint* colInd, __global const uint* clusters, __global const uint* n, __global const uint* numClusters, __global const float* totalEdgeWeight, __global float* modularity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + (uint)(1u));
    uint rowEnd = rowPtr[gidPlus1];
    float degreeAccum = as_float(0x00000000u);
    float intraWeightAccum = as_float(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      uint idxU32 = ((uint)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float edgeWeight = adjacencyValues[idxU32];
        uint neighbor = colInd[idxU32];
        uint neighborCluster = clusters[neighbor];
        float currentDegree = degreeAccum;
        float newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        float weightToAdd = ((sameCluster) ? (edgeWeight) : (as_float(0x00000000u)));
        float currentIntraWeight = intraWeightAccum;
        float newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    float degree = degreeAccum;
    float intraWeight = intraWeightAccum;
    float invTotalWeight = (as_float(0x3f800000u) / totalWeight);
    float eContrib = (intraWeight * invTotalWeight);
    float degreeNorm = (degree * invTotalWeight);
    float aContrib = (degreeNorm * degreeNorm);
    float contribution = (eContrib - aContrib);
    atomic_add(&modularity[0], contribution);
  }
}
