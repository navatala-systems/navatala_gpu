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

kernel void navatala_ml_softmax_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* _output [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  uint numIters = ((countVal + 255u) / 256u);
  float gsMax = as_type<float>(0xff7fc99eu);
  for (int itA = 0; itA < (int)(numIters); ++itA) {
    uint idxA = (lid + (((uint)(itA)) * 256u));
    if ((idxA < countVal)) {
      float xA = _input[idxA];
      gsMax = (((gsMax > xA)) ? (gsMax) : (xA));
    }
  }
  sdata[lid] = gsMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint maxStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = maxStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (((mine > other)) ? (mine) : (other));
      sdata[lid] = acc;
    }
    uint strideToHalve = maxStride;
    uint nextStride = (strideToHalve >> 1u);
    maxStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float maxVal = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsSum = as_type<float>(0x00000000u);
  for (int itB = 0; itB < (int)(numIters); ++itB) {
    uint idxB = (lid + (((uint)(itB)) * 256u));
    if ((idxB < countVal)) {
      float eB = exp((_input[idxB] - maxVal));
      gsSum = (gsSum + eB);
    }
  }
  sdata[lid] = gsSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sumStride = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = sumStride;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = sumStride;
    uint nextStride = (strideToHalve >> 1u);
    sumStride = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sumVal = sdata[0];
  for (int itC = 0; itC < (int)(numIters); ++itC) {
    uint idxC = (lid + (((uint)(itC)) * 256u));
    if ((idxC < countVal)) {
      float eC = exp((_input[idxC] - maxVal));
      _output[idxC] = (eC / sumVal);
    }
  }
}
