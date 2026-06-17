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

kernel void navatala_cfd_turbulence_dirichlet_face_elimination(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const int* constrainedMask [[buffer(2)]], device const float* constrainedValue [[buffer(3)]], device const int* nIntFaces [[buffer(4)]], device float* upper [[buffer(5)]], device float* lower [[buffer(6)]], device float* source [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if ((((int)(int(__gid.x))) >= nIntFaces[0])) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = neighbour[((int)(int(__gid.x)))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if (((oC | nC) == 0)) {
      return;
    } else {
      float up = upper[((int)(int(__gid.x)))];
      float lo = lower[((int)(int(__gid.x)))];
      if ((oC != 0)) {
        float cv = constrainedValue[o];
        atomic_fetch_add_explicit((device atomic_float*)(&source[n]), (-(lo * cv)), memory_order_relaxed);
      }
      if ((nC != 0)) {
        float cv = constrainedValue[n];
        atomic_fetch_add_explicit((device atomic_float*)(&source[o]), (-(up * cv)), memory_order_relaxed);
      }
      upper[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      lower[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}
