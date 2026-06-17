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

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_graph_compute_modularity_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* clusters [[buffer(3)]], device const uint* n [[buffer(4)]], device const uint* numClusters [[buffer(5)]], device const float* totalEdgeWeight [[buffer(6)]], device float* modularity [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float degreeAccum = as_type<float>(0x00000000u);
    float intraWeightAccum = as_type<float>(0x00000000u);
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
        float weightToAdd = ((sameCluster) ? (edgeWeight) : (as_type<float>(0x00000000u)));
        float currentIntraWeight = intraWeightAccum;
        float newIntraWeight = (currentIntraWeight + weightToAdd);
        intraWeightAccum = newIntraWeight;
      }
    }
    float degree = degreeAccum;
    float intraWeight = intraWeightAccum;
    float invTotalWeight = (as_type<float>(0x3f800000u) / totalWeight);
    float eContrib = (intraWeight * invTotalWeight);
    float degreeNorm = (degree * invTotalWeight);
    float aContrib = (degreeNorm * degreeNorm);
    float contribution = (eContrib - aContrib);
    atomic_fetch_add_explicit((device atomic_float*)(&modularity[0]), contribution, memory_order_relaxed);
  }
}
