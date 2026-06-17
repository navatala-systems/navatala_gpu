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

kernel void navatala_ml_reduction_min_f16(device const half* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device half* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsAcc = as_type<float>(0x7f7fc99eu);
  for (int it = 0; it < (int)(numIters); ++it) {
    uint idx = (lid + (((uint)(it)) * 256u));
    if ((idx < countVal)) {
      half raw = _input[idx];
      float v = ((float)(raw));
      gsAcc = (((gsAcc < v)) ? (gsAcc) : (v));
    }
  }
  sdata[lid] = gsAcc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint redStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = redStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine < other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = redStride;
    uint nextStride = (strideToHalve >> 1u);
    redStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float reduced = sdata[0];
    float nF = ((float)(countVal));
    float finalF = reduced;
    half outV = ((half)(finalF));
    result[0] = outV;
  }
}
