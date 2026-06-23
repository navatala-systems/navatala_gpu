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

static inline float gpu_atomic_min_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value < cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

static inline float gpu_atomic_max_float(device atomic_float* ptr, float value) {
  float cur = atomic_load_explicit(ptr, memory_order_relaxed);
  while (value > cur) {
    if (atomic_compare_exchange_weak_explicit(ptr, &cur, value,
        memory_order_relaxed, memory_order_relaxed)) { return value; }
  }
  return cur;
}

kernel void navatala_sparse_mst_boruvka_step_f32(device const uint* srcNodes [[buffer(0)]], device const uint* dstNodes [[buffer(1)]], device const float* weights [[buffer(2)]], device const uint* numEdges [[buffer(3)]], device const uint* numNodes [[buffer(4)]], device uint* components [[buffer(5)]], device uint* minEdgeIdx [[buffer(6)]], device float* minEdgeWeight [[buffer(7)]], device bool* mstEdges [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint tid = ((uint)(int(__gid.x)));
  if (tid < numEdges[0u]) {
    uint src = srcNodes[tid];
    uint dst = dstNodes[tid];
    uint srcRoot = components[src];
    uint dstRoot = components[dst];
    if (srcRoot != dstRoot) {
      float w = weights[tid];
      float currMin = minEdgeWeight[srcRoot];
      if (w < currMin) {
        gpu_atomic_min_float((device atomic_float*)(&minEdgeWeight[srcRoot]), w);
        minEdgeIdx[srcRoot] = tid;
      }
    }
  }
}
