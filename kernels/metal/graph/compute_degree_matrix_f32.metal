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

kernel void navatala_graph_compute_degree_matrix_f32(device const float* W [[buffer(0)]], device const uint* n [[buffer(1)]], device float* D [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint wgid = ((uint)(int(__tgid.x)));
  threadgroup float sdata[256];
  uint nVal = n[0];
  uint rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    uint rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    uint elemIdx = (rowBase + lid);
    if (colInBounds) {
      float wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = as_type<float>(0x00000000u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint degF32RedStride = 128u;
    for (int degF32RedStep = 0; degF32RedStep < (int)(8); ++degF32RedStep) {
      uint degF32Stride = degF32RedStride;
      if ((lid < degF32Stride)) {
        float other = sdata[(lid + degF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      uint degF32StrideToHalve = degF32RedStride;
      uint degF32NextStride = (degF32StrideToHalve >> 1u);
      degF32RedStride = degF32NextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if ((lid == 0u)) {
      float degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}
