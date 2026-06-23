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

kernel void navatala_ml_compute_m_s_e_f32(device const float* targets [[buffer(0)]], device const uint* sample_indices [[buffer(1)]], device const uint* n_node_samples [[buffer(2)]], device float* mse [[buffer(3)]], device float* mean_val [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint nSamples = n_node_samples[0];
  bool inBounds = (gid < nSamples);
  threadgroup float sdata[256];
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    float targetVal = targets[sampleIdx];
    sdata[lid] = targetVal;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mseF32_reductionStride = 128u;
  for (int mseF32_reductionStep = 0; mseF32_reductionStep < (int)(8); ++mseF32_reductionStep) {
    uint mseF32_stride = mseF32_reductionStride;
    if (lid < mseF32_stride) {
      uint otherIdx = (lid + mseF32_stride);
      float mine = sdata[lid];
      float other = sdata[otherIdx];
      float combined = (mine + other);
      sdata[lid] = combined;
    }
    uint mseF32_strideToHalve = mseF32_reductionStride;
    uint mseF32_nextStride = (mseF32_strideToHalve >> 1u);
    mseF32_reductionStride = mseF32_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  bool isFirst = (lid == 0u);
  if (isFirst) {
    float totalSum = sdata[0];
    float nSamplesF = ((float)(nSamples));
    float meanValue = (totalSum / nSamplesF);
    mean_val[0] = meanValue;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meanForDev = mean_val[0];
  if (inBounds) {
    uint sampleIdx2 = sample_indices[gid];
    float targetVal2 = targets[sampleIdx2];
    float dev = (targetVal2 - meanForDev);
    float devSq = (dev * dev);
    sdata[lid] = devSq;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mseF32b_reductionStride = 128u;
  for (int mseF32b_reductionStep = 0; mseF32b_reductionStep < (int)(8); ++mseF32b_reductionStep) {
    uint mseF32b_stride = mseF32b_reductionStride;
    if (lid < mseF32b_stride) {
      uint otherIdx2 = (lid + mseF32b_stride);
      float mine2 = sdata[lid];
      float other2 = sdata[otherIdx2];
      float combined2 = (mine2 + other2);
      sdata[lid] = combined2;
    }
    uint mseF32b_strideToHalve = mseF32b_reductionStride;
    uint mseF32b_nextStride = (mseF32b_strideToHalve >> 1u);
    mseF32b_reductionStride = mseF32b_nextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (isFirst) {
    float totalDevSq = sdata[0];
    float nSamplesF2 = ((float)(nSamples));
    float mseValue = (totalDevSq / nSamplesF2);
    mse[0] = mseValue;
  }
}
