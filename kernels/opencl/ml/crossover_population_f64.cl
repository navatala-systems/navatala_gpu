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
__kernel void navatala_ml_crossover_population_f64(__global const double* parent1, __global const double* parent2, __global const uint* crossoverPoints, __global const uint* nPairs, __global const uint* indivSize, __global double* offspring1, __global double* offspring2) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nPairsVal = nPairs[0];
  uint indivSizeVal = indivSize[0];
  uint totalGenes = (nPairsVal * indivSizeVal);
  bool inBounds = (gid < totalGenes);
  if (inBounds) {
    uint pairIdx = (gid / indivSizeVal);
    uint geneIdx = (gid % indivSizeVal);
    uint crossPt = crossoverPoints[pairIdx];
    double gene1 = parent1[gid];
    double gene2 = parent2[gid];
    bool beforeCrossover = (geneIdx < crossPt);
    double off1Gene = ((beforeCrossover) ? (gene1) : (gene2));
    double off2Gene = ((beforeCrossover) ? (gene2) : (gene1));
    offspring1[gid] = off1Gene;
    offspring2[gid] = off2Gene;
  }
}
