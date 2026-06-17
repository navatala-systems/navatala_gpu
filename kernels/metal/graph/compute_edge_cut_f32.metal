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

kernel void navatala_graph_compute_edge_cut_f32(device const float* adjacencyValues [[buffer(0)]], device const uint* rowPtr [[buffer(1)]], device const uint* colInd [[buffer(2)]], device const uint* clusters [[buffer(3)]], device const uint* n [[buffer(4)]], device float* edgeCut [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nVal = n[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    uint myCluster = clusters[gid];
    uint rowStart = rowPtr[gid];
    uint gidPlus1 = (gid + 1u);
    uint rowEnd = rowPtr[gidPlus1];
    float localCut = as_type<float>(0x00000000u);
    for (int edgeIdx = 0; edgeIdx < (int)(rowEnd); ++edgeIdx) {
      uint edgeIdxU = ((uint)(edgeIdx));
      bool edgeInRange = (edgeIdxU >= rowStart);
      if (edgeInRange) {
        uint neighbor = colInd[edgeIdxU];
        uint neighborCluster = clusters[neighbor];
        bool crossesBoundary = (myCluster != neighborCluster);
        if (crossesBoundary) {
          float edgeWeight = adjacencyValues[edgeIdxU];
          float currentCut = localCut;
          float newCut = (currentCut + edgeWeight);
          localCut = newCut;
        }
      }
    }
    float finalLocalCut = localCut;
    bool hasContribution = (finalLocalCut > as_type<float>(0x00000000u));
    if (hasContribution) {
      atomic_fetch_add_explicit((device atomic_float*)(&edgeCut[0]), finalLocalCut, memory_order_relaxed);
    }
  }
}
