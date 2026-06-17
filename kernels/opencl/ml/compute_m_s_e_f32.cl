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

__kernel void navatala_ml_compute_m_s_e_f32(__global const float* targets, __global const uint* sample_indices, __global const uint* n_node_samples, __global float* mse, __global float* mean_val) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint nSamples = n_node_samples[0];
  bool inBounds = (gid < nSamples);
  __local float sdata[256];
  if (inBounds) {
    uint sampleIdx = sample_indices[gid];
    float targetVal = targets[sampleIdx];
    sdata[lid] = targetVal;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint mseF32_reductionStride = (uint)(128u);
  for (int mseF32_reductionStep = 0; mseF32_reductionStep < (int)(8); ++mseF32_reductionStep) {
    uint mseF32_stride = mseF32_reductionStride;
    if ((lid < mseF32_stride)) {
      uint otherIdx = (lid + mseF32_stride);
      float mine = sdata[lid];
      float other = sdata[otherIdx];
      float combined = (mine + other);
      sdata[lid] = combined;
    }
    uint mseF32_strideToHalve = mseF32_reductionStride;
    uint mseF32_nextStride = (mseF32_strideToHalve >> (uint)(1u));
    mseF32_reductionStride = mseF32_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  bool isFirst = (lid == (uint)(0u));
  if (isFirst) {
    float totalSum = sdata[0];
    float nSamplesF = ((float)(nSamples));
    float meanValue = (totalSum / nSamplesF);
    mean_val[0] = meanValue;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float meanForDev = mean_val[0];
  if (inBounds) {
    uint sampleIdx2 = sample_indices[gid];
    float targetVal2 = targets[sampleIdx2];
    float dev = (targetVal2 - meanForDev);
    float devSq = (dev * dev);
    sdata[lid] = devSq;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint mseF32b_reductionStride = (uint)(128u);
  for (int mseF32b_reductionStep = 0; mseF32b_reductionStep < (int)(8); ++mseF32b_reductionStep) {
    uint mseF32b_stride = mseF32b_reductionStride;
    if ((lid < mseF32b_stride)) {
      uint otherIdx2 = (lid + mseF32b_stride);
      float mine2 = sdata[lid];
      float other2 = sdata[otherIdx2];
      float combined2 = (mine2 + other2);
      sdata[lid] = combined2;
    }
    uint mseF32b_strideToHalve = mseF32b_reductionStride;
    uint mseF32b_nextStride = (mseF32b_strideToHalve >> (uint)(1u));
    mseF32b_reductionStride = mseF32b_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (isFirst) {
    float totalDevSq = sdata[0];
    float nSamplesF2 = ((float)(nSamples));
    float mseValue = (totalDevSq / nSamplesF2);
    mse[0] = mseValue;
  }
}
