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

kernel void navatala_sparse_aggregate_size8_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const uint* strongMask [[buffer(3)]], device const uint* nRows [[buffer(4)]], device int* aggregateId [[buffer(5)]], device atomic_uint* nAggregates [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float bestVal_0 = as_type<float>(0x00000000u);
    int bestCol_0 = -1;
    float bestVal_1 = as_type<float>(0x00000000u);
    int bestCol_1 = -1;
    float bestVal_2 = as_type<float>(0x00000000u);
    int bestCol_2 = -1;
    float bestVal_3 = as_type<float>(0x00000000u);
    int bestCol_3 = -1;
    float bestVal_4 = as_type<float>(0x00000000u);
    int bestCol_4 = -1;
    float bestVal_5 = as_type<float>(0x00000000u);
    int bestCol_5 = -1;
    float bestVal_6 = as_type<float>(0x00000000u);
    int bestCol_6 = -1;
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int kk = (rs + j);
      int col = ((int)(colIdx[kk]));
      uint isStrong = strongMask[kk];
      if (isStrong == 1u) {
        float a = values[kk];
        float aAbs = abs(a);
        float oldVal_0 = bestVal_0;
        int oldCol_0 = bestCol_0;
        float oldVal_1 = bestVal_1;
        int oldCol_1 = bestCol_1;
        float oldVal_2 = bestVal_2;
        int oldCol_2 = bestCol_2;
        float oldVal_3 = bestVal_3;
        int oldCol_3 = bestCol_3;
        float oldVal_4 = bestVal_4;
        int oldCol_4 = bestCol_4;
        float oldVal_5 = bestVal_5;
        int oldCol_5 = bestCol_5;
        float oldVal_6 = bestVal_6;
        int oldCol_6 = bestCol_6;
        bestVal_0 = (((aAbs > oldVal_0)) ? (aAbs) : (oldVal_0));
        bestCol_0 = (((aAbs > oldVal_0)) ? (col) : (oldCol_0));
        bestVal_1 = (((aAbs > oldVal_0)) ? (oldVal_0) : ((((aAbs > oldVal_1)) ? (aAbs) : (oldVal_1))));
        bestCol_1 = (((aAbs > oldVal_0)) ? (oldCol_0) : ((((aAbs > oldVal_1)) ? (col) : (oldCol_1))));
        bestVal_2 = (((aAbs > oldVal_1)) ? (oldVal_1) : ((((aAbs > oldVal_2)) ? (aAbs) : (oldVal_2))));
        bestCol_2 = (((aAbs > oldVal_1)) ? (oldCol_1) : ((((aAbs > oldVal_2)) ? (col) : (oldCol_2))));
        bestVal_3 = (((aAbs > oldVal_2)) ? (oldVal_2) : ((((aAbs > oldVal_3)) ? (aAbs) : (oldVal_3))));
        bestCol_3 = (((aAbs > oldVal_2)) ? (oldCol_2) : ((((aAbs > oldVal_3)) ? (col) : (oldCol_3))));
        bestVal_4 = (((aAbs > oldVal_3)) ? (oldVal_3) : ((((aAbs > oldVal_4)) ? (aAbs) : (oldVal_4))));
        bestCol_4 = (((aAbs > oldVal_3)) ? (oldCol_3) : ((((aAbs > oldVal_4)) ? (col) : (oldCol_4))));
        bestVal_5 = (((aAbs > oldVal_4)) ? (oldVal_4) : ((((aAbs > oldVal_5)) ? (aAbs) : (oldVal_5))));
        bestCol_5 = (((aAbs > oldVal_4)) ? (oldCol_4) : ((((aAbs > oldVal_5)) ? (col) : (oldCol_5))));
        bestVal_6 = (((aAbs > oldVal_5)) ? (oldVal_5) : ((((aAbs > oldVal_6)) ? (aAbs) : (oldVal_6))));
        bestCol_6 = (((aAbs > oldVal_5)) ? (oldCol_5) : ((((aAbs > oldVal_6)) ? (col) : (oldCol_6))));
      }
    }
    int aggId = (((((((((false || (bestCol_0 >= 0)) || (bestCol_1 >= 0)) || (bestCol_2 >= 0)) || (bestCol_3 >= 0)) || (bestCol_4 >= 0)) || (bestCol_5 >= 0)) || (bestCol_6 >= 0))) ? (((((bestCol_6 >= 0) && (bestCol_6 < ((((bestCol_5 >= 0) && (bestCol_5 < ((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))) ? (bestCol_5) : (((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))))) ? (bestCol_6) : (((((bestCol_5 >= 0) && (bestCol_5 < ((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))) ? (bestCol_5) : (((((bestCol_4 >= 0) && (bestCol_4 < ((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))) ? (bestCol_4) : (((((bestCol_3 >= 0) && (bestCol_3 < ((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))) ? (bestCol_3) : (((((bestCol_2 >= 0) && (bestCol_2 < ((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))) ? (bestCol_2) : (((((bestCol_1 >= 0) && (bestCol_1 < ((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))) ? (bestCol_1) : (((((bestCol_0 >= 0) && (bestCol_0 < gid))) ? (bestCol_0) : (gid))))))))))))))) : (-1));
    aggregateId[gid] = aggId;
    if (aggId == gid) {
      uint _naBump = atomic_fetch_add_explicit(((device atomic_uint*)(&(nAggregates[0]))), 1u, memory_order_relaxed);
    }
  }
}
