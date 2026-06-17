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

kernel void navatala_sparse_idr_m_sync_fused_f32(device const float* P [[buffer(0)]], device const float* r [[buffer(1)]], device const float* Minv [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* s [[buffer(4)]], device float* c [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int k = int(__tgid.x);
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  float acc = as_type<float>(0x00000000u);
  for (int j = 0; j < (int)(S); ++j) {
    float pjr = (((gid < N)) ? ((P[(gid + (j * N))] * r[gid])) : (as_type<float>(0x00000000u)));
    float mij = Minv[(k + (j * S))];
    acc = (acc + (mij * pjr));
  }
  float warpSum = simd_sum(acc);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lid < 8)) {
    float sv = sdata[lid];
    float finalSum = simd_sum(sv);
    if ((lid == 0)) {
      c[k] = finalSum;
    }
  }
}
