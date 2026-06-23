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

kernel void navatala_sparse_cg_update_x_r_f32(device const float* p [[buffer(0)]], device const float* Ap [[buffer(1)]], device const float* alpha [[buffer(2)]], device const uint* n [[buffer(3)]], device float* x [[buffer(4)]], device float* r [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int N = ((int)(n[0]));
  if (gid < N) {
    float a = alpha[0];
    float pi = p[gid];
    float api = Ap[gid];
    float xi = x[gid];
    float ri = r[gid];
    x[gid] = (xi + (a * pi));
    r[gid] = (ri - (a * api));
  }
}
