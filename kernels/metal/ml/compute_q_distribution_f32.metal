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

kernel void navatala_ml_compute_q_distribution_f32(device const float* Y [[buffer(0)]], device const uint* nPoints [[buffer(1)]], device const uint* outputDims [[buffer(2)]], device float* Q [[buffer(3)]], device float* sumQ [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint n = nPoints[0];
  uint d = outputDims[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  float localQ = as_type<float>(0x00000000u);
  if (inBounds) {
    uint i = (gid / n);
    uint j = (gid % n);
    uint baseI = (i * d);
    uint baseJ = (j * d);
    float distSqAccum = as_type<float>(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = (baseI + kU32);
      uint idxJ = (baseJ + kU32);
      float yi_k = Y[idxI];
      float yj_k = Y[idxJ];
      float diff = (yi_k - yj_k);
      float diffSq = (diff * diff);
      float currentDistSq = distSqAccum;
      float newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    float distSq = distSqAccum;
    bool isDiag = (i == j);
    float onePlusDistSq = (as_type<float>(0x3f800000u) + distSq);
    float qVal = (as_type<float>(0x3f800000u) / onePlusDistSq);
    if (isDiag) {
      Q[gid] = as_type<float>(0x00000000u);
      sdata[lid] = as_type<float>(0x00000000u);
    } else {
      Q[gid] = qVal;
      sdata[lid] = qVal;
    }
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint qF32ReductionStride = 128u;
  for (int qF32ReductionStep = 0; qF32ReductionStep < (int)(8); ++qF32ReductionStep) {
    uint qF32Stride = qF32ReductionStride;
    if ((lid < qF32Stride)) {
      float qF32Other = sdata[(lid + qF32Stride)];
      float qF32Mine = sdata[lid];
      float qF32Sum = (qF32Mine + qF32Other);
      sdata[lid] = qF32Sum;
    }
    uint qF32StrideToHalve = qF32ReductionStride;
    uint qF32NextStride = (qF32StrideToHalve >> 1u);
    qF32ReductionStride = qF32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lid == 0u)) {
    float partialSum = sdata[0];
    float oldSum = sumQ[0];
    float newSum = (oldSum + partialSum);
    sumQ[0] = newSum;
  }
}
