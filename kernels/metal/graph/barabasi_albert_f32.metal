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

kernel void navatala_graph_barabasi_albert_f32(device const uint* cumDegree [[buffer(0)]], device const float* random [[buffer(1)]], device const float* randomWeights [[buffer(2)]], device const uint* newNode [[buffer(3)]], device const uint* totalDegree [[buffer(4)]], device const uint* m [[buffer(5)]], device const uint* numExistingNodes [[buffer(6)]], device uint* srcNodes [[buffer(7)]], device uint* dstNodes [[buffer(8)]], device float* weights [[buffer(9)]], device atomic_uint* edgeCount [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numM = m[0u];
  bool inBounds = (gid < numM);
  if (inBounds) {
    uint newNodeIdx = newNode[0u];
    uint totalDeg = totalDegree[0u];
    uint numExisting = numExistingNodes[0u];
    float r = random[gid];
    float totalDegF = ((float)(totalDeg));
    float targetDegF = (r * totalDegF);
    uint targetDeg = ((uint)(targetDegF));
    uint loAccum = 0u;
    uint hiAccum = numExisting;
    for (int _ = 0; _ < (int)(32u); ++_) {
      uint lo = loAccum;
      uint hi = hiAccum;
      bool stillSearching = (lo < hi);
      if (stillSearching) {
        uint mid = ((lo + hi) / 2u);
        uint cumVal = cumDegree[mid];
        bool goRight = (cumVal <= targetDeg);
        uint newLo = ((goRight) ? ((mid + 1u)) : (lo));
        uint newHi = ((goRight) ? (hi) : (mid));
        loAccum = newLo;
        hiAccum = newHi;
      }
    }
    uint finalLo = loAccum;
    uint targetNode = (((finalLo > 0u)) ? ((finalLo - 1u)) : (0u));
    float w = randomWeights[gid];
    uint slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(edgeCount[0u]))), 1u, memory_order_relaxed);
    srcNodes[slot] = newNodeIdx;
    dstNodes[slot] = targetNode;
    weights[slot] = w;
  }
}
