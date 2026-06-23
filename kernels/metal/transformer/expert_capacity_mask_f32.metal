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

kernel void navatala_transformer_expert_capacity_mask_f32(device const int* expertIndices [[buffer(0)]], device const uint* numTokens [[buffer(1)]], device const uint* numExperts [[buffer(2)]], device const uint* topK [[buffer(3)]], device const uint* capacity [[buffer(4)]], device float* mask [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nt = numTokens[0u];
  uint ne = numExperts[0u];
  uint k = topK[0u];
  uint cap = capacity[0u];
  threadgroup uint expertCounts[256];
  uint lid = ((uint)(int(__tid.x)));
  if (lid < ne) {
    expertCounts[lid] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint totalElements = (nt * k);
  bool valid = (gid < totalElements);
  if (valid) {
    int expertIdx = expertIndices[gid];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertIdxValid = (expertIdxU32 < ne);
    if (expertIdxValid) {
      uint prevCount = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounts[expertIdxU32]))), 1u, memory_order_relaxed);
      bool withinCapacity = (prevCount < cap);
      float maskVal = ((withinCapacity) ? (as_type<float>(0x3f800000u)) : (as_type<float>(0x00000000u)));
      mask[gid] = maskVal;
    } else {
      mask[gid] = as_type<float>(0x00000000u);
    }
  }
}
