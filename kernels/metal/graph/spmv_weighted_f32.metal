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

kernel void navatala_graph_spmv_weighted_f32(device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* x [[buffer(3)]], device const uint* numVertices [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint numV = numVertices[0];
  if ((gid < numV)) {
    uint base = offsets[gid];
    uint endv = offsets[(gid + 1u)];
    uint rowlen = (endv - base);
    float acc = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(rowlen); ++k) {
      uint eidx = (base + ((uint)(k)));
      uint col = indices[eidx];
      float w = weights[eidx];
      float xv = x[col];
      acc = (acc + (w * xv));
    }
    y[gid] = acc;
  }
}
