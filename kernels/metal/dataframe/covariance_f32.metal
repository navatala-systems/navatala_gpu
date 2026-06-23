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

kernel void navatala_dataframe_covariance_f32(device const float* inputX [[buffer(0)]], device const float* inputY [[buffer(1)]], device const uint* count [[buffer(2)]], device const float* meanX [[buffer(3)]], device const float* meanY [[buffer(4)]], device float* covariance [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float sdata[256];
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint countVal = count[0];
  float meanXVal = meanX[0];
  float meanYVal = meanY[0];
  bool inBounds = (gid < countVal);
  float x = inputX[gid];
  float y = inputY[gid];
  float dx = (x - meanXVal);
  float dy = (y - meanYVal);
  float product = (dx * dy);
  if (inBounds) {
    sdata[lid] = product;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint covF32_reductionStride = 128u;
  for (int covF32_reductionStep = 0; covF32_reductionStep < (int)(8); ++covF32_reductionStep) {
    uint covF32_stride = covF32_reductionStride;
    if (lid < covF32_stride) {
      float covF32_other = sdata[(lid + covF32_stride)];
      float covF32_mine = sdata[lid];
      float covF32_sum = (covF32_mine + covF32_other);
      sdata[lid] = covF32_sum;
    }
    uint covF32_strideToHalve = covF32_reductionStride;
    uint covF32_nextStride = (covF32_strideToHalve >> 1u);
    covF32_reductionStride = covF32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalSum = sdata[0u];
    float countFloat = ((float)(countVal));
    float covVal = (totalSum / countFloat);
    covariance[0u] = covVal;
  }
}
