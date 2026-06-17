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

kernel void navatala_ml_layernorm_f16(device const half* x [[buffer(0)]], device const half* gamma [[buffer(1)]], device const half* beta [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device half* _output [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = as_type<float>(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if (((lid + (((uint)(itM)) * 256u)) < countVal)) {
      gsM = (gsM + ((float)(x[(lid + (((uint)(itM)) * 256u))])));
    }
  }
  sdata[lid] = gsM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gsV = as_type<float>(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if (((lid + (((uint)(itV)) * 256u)) < countVal)) {
      gsV = (gsV + ((((float)(x[(lid + (((uint)(itV)) * 256u))])) - mean) * (((float)(x[(lid + (((uint)(itV)) * 256u))])) - mean)));
    }
  }
  sdata[lid] = gsV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    uint idxW = (lid + (((uint)(itW)) * 256u));
    if ((idxW < countVal)) {
      float xnW = ((((float)(x[idxW])) - mean) / denom);
      float outF = ((((float)(gamma[idxW])) * xnW) + ((float)(beta[idxW])));
      _output[idxW] = ((half)(outF));
    }
  }
}
