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

kernel void navatala_dataframe_mape_f32(device const float* y_true [[buffer(0)]], device const float* y_pred [[buffer(1)]], device const uint* n [[buffer(2)]], device float* result [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = n[0];
  bool inBounds = (gid < countVal);
  float trueVal = y_true[gid];
  float predVal = y_pred[gid];
  float diff = (trueVal - predVal);
  float absErr = abs(diff);
  float absTrueVal = abs(trueVal);
  float pctErr = (absErr / absTrueVal);
  if (inBounds) {
    sdata[lid] = pctErr;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mapeF32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint mapeF32_stride = mapeF32_reductionStride;
    if (lid < mapeF32_stride) {
      uint mapeF32_partnerIdx = (lid + mapeF32_stride);
      float other = sdata[mapeF32_partnerIdx];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint mapeF32_strideToHalve = mapeF32_reductionStride;
    uint mapeF32_nextStride = (mapeF32_strideToHalve >> 1u);
    mapeF32_reductionStride = mapeF32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint zeroU32 = ((uint)(0));
  if (lid == zeroU32) {
    float totalSum = sdata[0];
    float countFloat = ((float)(countVal));
    float avgPctErr = (totalSum / countFloat);
    float mapeVal = (as_type<float>(0x42c80000u) * avgPctErr);
    result[0] = mapeVal;
  }
}
