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

kernel void navatala_sparse_csr_symmetry_check_f32(device const uint* rowPtr [[buffer(0)]], device const uint* colIdx [[buffer(1)]], device const float* values [[buffer(2)]], device const float* tol [[buffer(3)]], device const uint* nRows [[buffer(4)]], device uint* asymmetricFlags [[buffer(5)]], device atomic_uint* isSymmetric [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if ((col < gid)) {
      } else {
        if ((col > gid)) {
          float a = values[k];
          int colRs = ((int)(rowPtr[col]));
          int colRe = ((int)(rowPtr[(col + 1)]));
          uint found = 0u;
          for (int jj = 0; jj < (int)((colRe - colRs)); ++jj) {
            int kk = (colRs + jj);
            int cc = ((int)(colIdx[kk]));
            if ((cc == gid)) {
              float at = values[kk];
              float tolVal = tol[0];
              float diff = abs((a - at));
              if ((diff > tolVal)) {
                asymmetricFlags[gid] = 1u;
                uint _aod5 = atomic_fetch_and_explicit(((device atomic_uint*)(&(isSymmetric[0]))), 0u, memory_order_relaxed);
              }
            }
          }
        }
      }
    }
  }
}
