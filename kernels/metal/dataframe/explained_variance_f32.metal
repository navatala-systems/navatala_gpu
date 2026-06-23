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

kernel void navatala_dataframe_explained_variance_f32(device const float* y_true [[buffer(0)]], device const float* y_pred [[buffer(1)]], device const float* mean_residuals [[buffer(2)]], device const float* mean_true [[buffer(3)]], device const uint* n [[buffer(4)]], device float* result [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata_var_res[256];
  threadgroup float sdata_var_true[256];
  uint countVal = n[0];
  float meanResVal = mean_residuals[0];
  float meanTrueVal = mean_true[0];
  bool inBounds = (gid < countVal);
  float trueVal = y_true[gid];
  float predVal = y_pred[gid];
  float residual = (trueVal - predVal);
  float resDiff = (residual - meanResVal);
  float sqResDiff = (resDiff * resDiff);
  float trueDiff = (trueVal - meanTrueVal);
  float sqTrueDiff = (trueDiff * trueDiff);
  if (inBounds) {
    sdata_var_res[lid] = sqResDiff;
    sdata_var_true[lid] = sqTrueDiff;
  } else {
    sdata_var_res[lid] = as_type<float>(0x00000000u);
    sdata_var_true[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint evF32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint evF32_stride = evF32_reductionStride;
    if (lid < evF32_stride) {
      uint evF32_partnerIdx = (lid + evF32_stride);
      float otherRes = sdata_var_res[evF32_partnerIdx];
      float mineRes = sdata_var_res[lid];
      float sumRes = (mineRes + otherRes);
      sdata_var_res[lid] = sumRes;
      float otherTrue = sdata_var_true[evF32_partnerIdx];
      float mineTrue = sdata_var_true[lid];
      float sumTrue = (mineTrue + otherTrue);
      sdata_var_true[lid] = sumTrue;
    }
    uint evF32_strideToHalve = evF32_reductionStride;
    uint evF32_nextStride = (evF32_strideToHalve >> 1u);
    evF32_reductionStride = evF32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint zeroU32 = ((uint)(0));
  if (lid == zeroU32) {
    float sumSqRes = sdata_var_res[0];
    float sumSqTrue = sdata_var_true[0];
    float countFloat = ((float)(countVal));
    float varRatio = (sumSqRes / sumSqTrue);
    float evVal = (as_type<float>(0x3f800000u) - varRatio);
    result[0] = evVal;
  }
}
