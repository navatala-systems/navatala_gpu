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

kernel void navatala_ml_layernorm_backward_f16(device const half* x [[buffer(0)]], device const half* dy [[buffer(1)]], device const half* gamma [[buffer(2)]], device const uint* count [[buffer(3)]], device const float* eps [[buffer(4)]], device half* dx [[buffer(5)]], device half* dgamma [[buffer(6)]], device half* dbeta [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gM = as_type<float>(0x00000000u);
  for (int itm = 0; itm < (int)(numIters); ++itm) {
    if ((lid + (((uint)(itm)) * 256u)) < countVal) {
      gM = (gM + ((float)(x[(lid + (((uint)(itm)) * 256u))])));
    }
  }
  sdata[lid] = gM;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = mStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = mStr;
    uint nextStride = (strideToHalve >> 1u);
    mStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gV = as_type<float>(0x00000000u);
  for (int itv = 0; itv < (int)(numIters); ++itv) {
    if ((lid + (((uint)(itv)) * 256u)) < countVal) {
      gV = (gV + ((((float)(x[(lid + (((uint)(itv)) * 256u))])) - mean) * (((float)(x[(lid + (((uint)(itv)) * 256u))])) - mean)));
    }
  }
  sdata[lid] = gV;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint vStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = vStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = vStr;
    uint nextStride = (strideToHalve >> 1u);
    vStr = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float var = (sdata[0] / n);
  float std = sqrt((var + epsVal));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g1 = as_type<float>(0x00000000u);
  for (int it1 = 0; it1 < (int)(numIters); ++it1) {
    if ((lid + (((uint)(it1)) * 256u)) < countVal) {
      g1 = (g1 + (((float)(dy[(lid + (((uint)(it1)) * 256u))])) * ((float)(gamma[(lid + (((uint)(it1)) * 256u))]))));
    }
  }
  sdata[lid] = g1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s1Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s1Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s1Str;
    uint nextStride = (strideToHalve >> 1u);
    s1Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m1 = (sdata[0] / n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float g2 = as_type<float>(0x00000000u);
  for (int it2 = 0; it2 < (int)(numIters); ++it2) {
    if ((lid + (((uint)(it2)) * 256u)) < countVal) {
      g2 = (g2 + ((((float)(dy[(lid + (((uint)(it2)) * 256u))])) * ((float)(gamma[(lid + (((uint)(it2)) * 256u))]))) * ((((float)(x[(lid + (((uint)(it2)) * 256u))])) - mean) / std)));
    }
  }
  sdata[lid] = g2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s2Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s2Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s2Str;
    uint nextStride = (strideToHalve >> 1u);
    s2Str = nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m2 = (sdata[0] / n);
  for (int itw = 0; itw < (int)(numIters); ++itw) {
    uint idw = (lid + (((uint)(itw)) * 256u));
    if (idw < countVal) {
      float xhatw = ((((float)(x[idw])) - mean) / std);
      float dyvw = ((float)(dy[idw]));
      float dxhw = (dyvw * ((float)(gamma[idw])));
      float dxFw = (((dxhw - m1) - (xhatw * m2)) / std);
      dx[idw] = ((half)(dxFw));
      dgamma[idw] = ((half)((dyvw * xhatw)));
      dbeta[idw] = ((half)(dyvw));
    }
  }
}
