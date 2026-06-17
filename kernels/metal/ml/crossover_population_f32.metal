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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_crossover_population_f32(device const float* parent1 [[buffer(0)]], device const float* parent2 [[buffer(1)]], device const uint* crossoverPoints [[buffer(2)]], device const uint* nPairs [[buffer(3)]], device const uint* indivSize [[buffer(4)]], device float* offspring1 [[buffer(5)]], device float* offspring2 [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nPairsVal = nPairs[0];
  uint indivSizeVal = indivSize[0];
  uint totalGenes = (nPairsVal * indivSizeVal);
  bool inBounds = (gid < totalGenes);
  if (inBounds) {
    uint pairIdx = (gid / indivSizeVal);
    uint geneIdx = (gid % indivSizeVal);
    uint crossPt = crossoverPoints[pairIdx];
    float gene1 = parent1[gid];
    float gene2 = parent2[gid];
    bool beforeCrossover = (geneIdx < crossPt);
    float off1Gene = ((beforeCrossover) ? (gene1) : (gene2));
    float off2Gene = ((beforeCrossover) ? (gene2) : (gene1));
    offspring1[gid] = off1Gene;
    offspring2[gid] = off2Gene;
  }
}
