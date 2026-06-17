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

kernel void navatala_linalg_reduce_max_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    half xf16 = _input[gid];
    float xf32 = ((float)(xf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = as_type<float>(0xff7ffffdu);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxF16ReductionStride = 128u;
  for (int maxF16ReductionStep = 0; maxF16ReductionStep < (int)(8); ++maxF16ReductionStep) {
    uint maxF16Stride = maxF16ReductionStride;
    if ((lid < maxF16Stride)) {
      float other = sdata[(lid + maxF16Stride)];
      float mine = sdata[lid];
      bool isGreater = (mine > other);
      float maxVal = ((isGreater) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    uint maxF16StrideToHalve = maxF16ReductionStride;
    uint maxF16NextStride = (maxF16StrideToHalve >> 1u);
    maxF16ReductionStride = maxF16NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float maxResult = sdata[0];
    result[0] = maxResult;
  }
}
