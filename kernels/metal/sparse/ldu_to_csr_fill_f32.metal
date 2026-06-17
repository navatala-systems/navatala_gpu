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

kernel void navatala_sparse_ldu_to_csr_fill_f32(device const uint* owner [[buffer(0)]], device const uint* neighbour [[buffer(1)]], device const float* upper [[buffer(2)]], device const float* lower [[buffer(3)]], device const uint* rowPtr [[buffer(4)]], device const uint* nInternalFaces [[buffer(5)]], device const uint* nCells [[buffer(6)]], device uint* colInd [[buffer(7)]], device float* values [[buffer(8)]], device atomic_uint* insertIdx [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int face = int(__gid.x);
  int nFaces = ((int)(nInternalFaces[0]));
  if ((face < nFaces)) {
    int own = ((int)(owner[face]));
    int nbr = ((int)(neighbour[face]));
    uint posOwn = atomic_fetch_add_explicit(((device atomic_uint*)(&(insertIdx[own]))), 1u, memory_order_relaxed);
    int offOwn = (((int)(rowPtr[own])) + ((int)(posOwn)));
    colInd[offOwn] = neighbour[face];
    values[offOwn] = upper[face];
    uint posNbr = atomic_fetch_add_explicit(((device atomic_uint*)(&(insertIdx[nbr]))), 1u, memory_order_relaxed);
    int offNbr = (((int)(rowPtr[nbr])) + ((int)(posNbr)));
    colInd[offNbr] = owner[face];
    values[offNbr] = lower[face];
  }
}
