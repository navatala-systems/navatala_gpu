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

kernel void navatala_graph_rmat_generator_f32(device const float* random [[buffer(0)]], device const float* randomWeights [[buffer(1)]], device const float* a [[buffer(2)]], device const float* b [[buffer(3)]], device const float* c [[buffer(4)]], device const float* d [[buffer(5)]], device const uint* n [[buffer(6)]], device const uint* scale [[buffer(7)]], device const uint* numEdges [[buffer(8)]], device uint* srcNodes [[buffer(9)]], device uint* dstNodes [[buffer(10)]], device float* weights [[buffer(11)]], device atomic_uint* edgeCount [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numE = numEdges[0u];
  bool inBounds = (gid < numE);
  if (inBounds) {
    float probA = a[0u];
    float probB = b[0u];
    float probC = c[0u];
    uint scaleVal = scale[0u];
    float ab = (probA + probB);
    float abc = (ab + probC);
    uint srcAccum = 0u;
    uint dstAccum = 0u;
    for (int level = 0; level < (int)(scaleVal); ++level) {
      uint levelU32 = ((uint)(level));
      uint randIdx = ((gid * scaleVal) + levelU32);
      float r = random[randIdx];
      uint bitPos = ((scaleVal - levelU32) - 1u);
      uint bitMask = (1u << bitPos);
      bool inQ01 = (r >= probA);
      bool inQ10 = (r >= ab);
      bool inQ11 = (r >= abc);
      uint srcBit = ((inQ10) ? (bitMask) : (0u));
      uint currentSrc = srcAccum;
      uint newSrc = (currentSrc | srcBit);
      srcAccum = newSrc;
      bool dstBitCond = ((inQ01 && (!inQ10)) || inQ11);
      uint dstBit = ((dstBitCond) ? (bitMask) : (0u));
      uint currentDst = dstAccum;
      uint newDst = (currentDst | dstBit);
      dstAccum = newDst;
    }
    uint src = srcAccum;
    uint dst = dstAccum;
    float w = randomWeights[gid];
    uint slot = atomic_fetch_add_explicit(((device atomic_uint*)(&(edgeCount[0u]))), 1u, memory_order_relaxed);
    srcNodes[slot] = src;
    dstNodes[slot] = dst;
    weights[slot] = w;
  }
}
