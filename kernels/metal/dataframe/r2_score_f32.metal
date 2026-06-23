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

kernel void navatala_dataframe_r2_score_f32(device const float* y_true [[buffer(0)]], device const float* y_pred [[buffer(1)]], device const float* y_mean [[buffer(2)]], device const uint* n [[buffer(3)]], device float* result [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata_res[256];
  threadgroup float sdata_tot[256];
  uint countVal = n[0];
  float meanVal = y_mean[0];
  bool inBounds = (gid < countVal);
  float trueVal = y_true[gid];
  float predVal = y_pred[gid];
  float diffRes = (trueVal - predVal);
  float sqRes = (diffRes * diffRes);
  float diffTot = (trueVal - meanVal);
  float sqTot = (diffTot * diffTot);
  if (inBounds) {
    sdata_res[lid] = sqRes;
    sdata_tot[lid] = sqTot;
  } else {
    sdata_res[lid] = as_type<float>(0x00000000u);
    sdata_tot[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint r2F32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint r2F32_stride = r2F32_reductionStride;
    if (lid < r2F32_stride) {
      uint r2F32_partnerIdx = (lid + r2F32_stride);
      float otherRes = sdata_res[r2F32_partnerIdx];
      float mineRes = sdata_res[lid];
      float sumRes = (mineRes + otherRes);
      sdata_res[lid] = sumRes;
      float otherTot = sdata_tot[r2F32_partnerIdx];
      float mineTot = sdata_tot[lid];
      float sumTot = (mineTot + otherTot);
      sdata_tot[lid] = sumTot;
    }
    uint r2F32_strideToHalve = r2F32_reductionStride;
    uint r2F32_nextStride = (r2F32_strideToHalve >> 1u);
    r2F32_reductionStride = r2F32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint zeroU32 = ((uint)(0));
  if (lid == zeroU32) {
    float ssRes = sdata_res[0];
    float ssTot = sdata_tot[0];
    float ratio = (ssRes / ssTot);
    float r2Val = (as_type<float>(0x3f800000u) - ratio);
    result[0] = r2Val;
  }
}
