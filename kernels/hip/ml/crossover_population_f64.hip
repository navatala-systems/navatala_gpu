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
extern "C" __global__ void navatala_ml_crossover_population_f64(const double* parent1, const double* parent2, const unsigned int* crossoverPoints, const unsigned int* nPairs, const unsigned int* indivSize, double* offspring1, double* offspring2) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nPairsVal = nPairs[0];
  unsigned int indivSizeVal = indivSize[0];
  unsigned int totalGenes = (nPairsVal * indivSizeVal);
  bool inBounds = (gid < totalGenes);
  if (inBounds) {
    unsigned int pairIdx = (gid / indivSizeVal);
    unsigned int geneIdx = (gid % indivSizeVal);
    unsigned int crossPt = crossoverPoints[pairIdx];
    double gene1 = parent1[gid];
    double gene2 = parent2[gid];
    bool beforeCrossover = (geneIdx < crossPt);
    double off1Gene = ((beforeCrossover) ? (gene1) : (gene2));
    double off2Gene = ((beforeCrossover) ? (gene2) : (gene1));
    offspring1[gid] = off1Gene;
    offspring2[gid] = off2Gene;
  }
}
