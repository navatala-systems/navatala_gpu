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
extern "C" __global__ void navatala_graph_compute_modularity_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* clusters, const unsigned int* n, const unsigned int* numClusters, const double* totalEdgeWeight, double* modularity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int myCluster = clusters[gid];
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double degreeAccum = __longlong_as_double(0x0000000000000000ull);
    double intraWeightAccum = __longlong_as_double(0x0000000000000000ull);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double edgeWeight = adjacencyValues[idxU32];
        unsigned int neighbor = colInd[idxU32];
        unsigned int neighborCluster = clusters[neighbor];
        double currentDegree = degreeAccum;
        double newDegree = (currentDegree + edgeWeight);
        degreeAccum = newDegree;
        bool sameCluster = (neighborCluster == myCluster);
        double weightToAdd = ((sameCluster) ? (edgeWeight) : (__longlong_as_double(0x0000000000000000ull)));
        double currentIntraWeight = intraWeightAccum;
        double newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    double degree = degreeAccum;
    double intraWeight = intraWeightAccum;
    double invTotalWeight = (__longlong_as_double(0x3ff0000000000000ull) / totalWeight);
    double eContrib = (intraWeight * invTotalWeight);
    double degreeNorm = (degree * invTotalWeight);
    double aContrib = (degreeNorm * degreeNorm);
    double contribution = (eContrib - aContrib);
    atomicAdd(&modularity[0], contribution);
  }
}
