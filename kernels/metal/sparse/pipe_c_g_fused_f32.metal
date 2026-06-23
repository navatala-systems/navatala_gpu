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

kernel void navatala_sparse_pipe_c_g_fused_f32(device const float* p [[buffer(0)]], device const float* Ap [[buffer(1)]], device const float* diagInv [[buffer(2)]], device const float* alpha [[buffer(3)]], device const uint* n [[buffer(4)]], device float* x [[buffer(5)]], device float* r [[buffer(6)]], device float* z [[buffer(7)]], device float* partials [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float a = alpha[0];
  bool inBounds = (gid < N);
  if (inBounds) {
    float pi = p[gid];
    float api = Ap[gid];
    float xi = x[gid];
    float ri = r[gid];
    float rnew = (ri - (a * api));
    x[gid] = (xi + (a * pi));
    r[gid] = rnew;
    float di = diagInv[gid];
    z[gid] = (di * rnew);
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
      partials[grpId] = finalSum;
    }
  }
}
