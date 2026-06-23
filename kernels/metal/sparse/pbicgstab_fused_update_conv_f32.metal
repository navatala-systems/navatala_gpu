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

kernel void navatala_sparse_pbicgstab_fused_update_conv_f32(device const float* phat [[buffer(0)]], device const float* shat [[buffer(1)]], device const float* s [[buffer(2)]], device const float* t [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* omega [[buffer(5)]], device const uint* n [[buffer(6)]], device float* x [[buffer(7)]], device float* r [[buffer(8)]], device float* rrPartials [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  if (inBounds) {
    float a = alpha[0];
    float om = omega[0];
    float ph = phat[gid];
    float sh = shat[gid];
    float si = s[gid];
    float ti = t[gid];
    float xi = x[gid];
    float ri = (si - (om * ti));
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = ri;
  }
  float sq = ((inBounds) ? ((r[gid] * r[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if (lid == 0) {
      int grpId = int(__tgid.x);
      rrPartials[grpId] = finalSum;
    }
  }
}
