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

kernel void navatala_linalg_norm_linf_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint gid = ((uint)(int(__gid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  float val = abs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint linfF32RedStride = 128u;
  for (int linfF32RedStep = 0; linfF32RedStep < (int)(8); ++linfF32RedStep) {
    uint linfF32Stride = linfF32RedStride;
    if (lid < linfF32Stride) {
      float other = sdata[(lid + linfF32Stride)];
      float mine = sdata[lid];
      bool mineGtOther = (mine > other);
      float maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint linfF32StrideToHalve = linfF32RedStride;
    uint linfF32NextStride = (linfF32StrideToHalve >> 1u);
    linfF32RedStride = linfF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    result[0] = sdata[0];
  }
}
