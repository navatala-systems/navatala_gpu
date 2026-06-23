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

kernel void navatala_sparse_subspace_convergence_f32(device const float* AX [[buffer(0)]], device const float* X [[buffer(1)]], device const float* lambdas [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* k [[buffer(4)]], device const float* tolerance [[buffer(5)]], device float* residNorms [[buffer(6)]], device atomic_uint* nConverged [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  int kVal = ((int)(k[0]));
  if (gid < kVal) {
    float norm2 = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(N); ++i) {
      float ax = AX[((gid * N) + i)];
      float x = X[((gid * N) + i)];
      float lam = lambdas[gid];
      norm2 = (norm2 + ((ax - (lam * x)) * (ax - (lam * x))));
    }
    residNorms[gid] = norm2;
    float tol = tolerance[0];
    if (norm2 < (tol * tol)) {
      uint _aod1 = atomic_fetch_add_explicit(((device atomic_uint*)(&(nConverged[0]))), 1u, memory_order_relaxed);
    }
  }
}
