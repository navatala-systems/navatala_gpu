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

kernel void navatala_sparse_compute_m_i_s_f32(device const uint* strongRowPtr [[buffer(0)]], device const uint* strongColIdx [[buffer(1)]], device const uint* randomPriority [[buffer(2)]], device const uint* nCells [[buffer(3)]], device int* aggregateId [[buffer(4)]], device atomic_uint* nUndecided [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int cell = int(__gid.x);
  int N = ((int)(nCells[0]));
  if ((cell < N)) {
    int myAgg = aggregateId[cell];
    if ((myAgg == -1)) {
      uint myPri = randomPriority[cell];
      int rs = ((int)(strongRowPtr[cell]));
      int re = ((int)(strongRowPtr[(cell + 1)]));
      int isMax = 1;
      for (int j = 0; j < (int)((re - rs)); ++j) {
        int k = (rs + j);
        int nbr = ((int)(strongColIdx[k]));
        int nbrAgg = aggregateId[nbr];
        if ((nbrAgg == -1)) {
          uint nbrPri = randomPriority[nbr];
          if (((nbrPri > myPri) || ((nbrPri == myPri) && (nbr < cell)))) {
            isMax = 0;
          }
        }
      }
      if ((isMax == 1)) {
        aggregateId[cell] = cell;
      } else {
        int assigned = 0;
        for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
          if ((assigned == 0)) {
            int k2 = (rs + j2);
            int nbr2 = ((int)(strongColIdx[k2]));
            int nbrAgg2 = aggregateId[nbr2];
            if ((nbrAgg2 >= 0)) {
              aggregateId[cell] = nbrAgg2;
              assigned = 1;
            }
          }
        }
        if ((assigned == 0)) {
          uint _u = atomic_fetch_add_explicit(((device atomic_uint*)(&(nUndecided[0]))), 1u, memory_order_relaxed);
        }
      }
    }
  }
}
