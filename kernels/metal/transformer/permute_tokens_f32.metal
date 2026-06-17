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

kernel void navatala_transformer_permute_tokens_f32(device const float* hiddenStates [[buffer(0)]], device const int* expertIndices [[buffer(1)]], device const uint* expertOffsets [[buffer(2)]], device atomic_uint* expertCounters [[buffer(3)]], device const uint* numTokens [[buffer(4)]], device const uint* hiddenSize [[buffer(5)]], device const uint* numExperts [[buffer(6)]], device float* permutedStates [[buffer(7)]], device int* tokenIndices [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint tokenIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  uint ne = numExperts[0u];
  threadgroup uint sharedDestTokenIdx[1];
  bool tokenValid = (tokenIdx < nt);
  if (tokenValid) {
    int expertIdx = expertIndices[tokenIdx];
    uint expertIdxU32 = ((uint)(expertIdx));
    bool expertValid = (expertIdxU32 < ne);
    if (expertValid) {
      if ((lid == 0u)) {
        uint localPosition = atomic_fetch_add_explicit(((device atomic_uint*)(&(expertCounters[expertIdxU32]))), 1u, memory_order_relaxed);
        uint expertOffset = expertOffsets[expertIdxU32];
        uint destTokenIdx = (expertOffset + localPosition);
        sharedDestTokenIdx[0u] = destTokenIdx;
        int tokenIdxI32 = ((int)(tokenIdx));
        tokenIndices[destTokenIdx] = tokenIdxI32;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint destTokenIdx = sharedDestTokenIdx[0u];
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((tokenIdx * hs) + hIdx);
          uint dstIdx = ((destTokenIdx * hs) + hIdx);
          float val = hiddenStates[srcIdx];
          permutedStates[dstIdx] = val;
        }
      }
    }
  }
}
