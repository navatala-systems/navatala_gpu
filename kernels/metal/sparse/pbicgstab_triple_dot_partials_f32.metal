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

kernel void navatala_sparse_pbicgstab_triple_dot_partials_f32(device const float* r0hat [[buffer(0)]], device const float* r [[buffer(1)]], device const float* v [[buffer(2)]], device const float* s [[buffer(3)]], device const uint* n [[buffer(4)]], device float* partials [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  float h0 = ((inBounds) ? (r0hat[gid]) : (as_type<float>(0x00000000u)));
  float d0 = ((inBounds) ? ((h0 * r[gid])) : (as_type<float>(0x00000000u)));
  float d1 = ((inBounds) ? ((h0 * v[gid])) : (as_type<float>(0x00000000u)));
  float d2 = ((inBounds) ? ((h0 * s[gid])) : (as_type<float>(0x00000000u)));
  float ws0 = simd_sum(d0);
  float ws1 = simd_sum(d1);
  float ws2 = simd_sum(d2);
  threadgroup float sd0[32];
  threadgroup float sd1[32];
  threadgroup float sd2[32];
  int lane = int(__lane);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sd0[warpIdx] = ws0;
    sd1[warpIdx] = ws1;
    sd2[warpIdx] = ws2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lid < 8)) {
    int grpId = int(__tgid.x);
    int base = (grpId * 3);
    float f0 = simd_sum(sd0[lid]);
    float f1 = simd_sum(sd1[lid]);
    float f2 = simd_sum(sd2[lid]);
    if ((lid == 0)) {
      partials[base] = f0;
      partials[(base + 1)] = f1;
      partials[(base + 2)] = f2;
    }
  }
}
