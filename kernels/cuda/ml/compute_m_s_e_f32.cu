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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_compute_m_s_e_f32(const float* targets, const unsigned int* sample_indices, const unsigned int* n_node_samples, float* mse, float* mean_val) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nSamples = n_node_samples[0];
  bool inBounds = (gid < nSamples);
  __shared__ float sdata[256];
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    float targetVal = targets[sampleIdx];
    sdata[lid] = targetVal;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int mseF32_reductionStride = 128u;
  for (int mseF32_reductionStep = 0; mseF32_reductionStep < (int)(8); ++mseF32_reductionStep) {
    unsigned int mseF32_stride = mseF32_reductionStride;
    if (lid < mseF32_stride) {
      unsigned int otherIdx = (lid + mseF32_stride);
      float mine = sdata[lid];
      float other = sdata[otherIdx];
      float combined = (mine + other);
      sdata[lid] = combined;
    }
    unsigned int mseF32_strideToHalve = mseF32_reductionStride;
    unsigned int mseF32_nextStride = (mseF32_strideToHalve >> 1u);
    mseF32_reductionStride = mseF32_nextStride;
    __syncthreads();
  }
  bool isFirst = (lid == 0u);
  if (isFirst) {
    float totalSum = sdata[0];
    float nSamplesF = ((float)(nSamples));
    float meanValue = (totalSum / nSamplesF);
    mean_val[0] = meanValue;
  }
  __syncthreads();
  float meanForDev = mean_val[0];
  if (inBounds) {
    unsigned int sampleIdx2 = sample_indices[gid];
    float targetVal2 = targets[sampleIdx2];
    float dev = (targetVal2 - meanForDev);
    float devSq = (dev * dev);
    sdata[lid] = devSq;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int mseF32b_reductionStride = 128u;
  for (int mseF32b_reductionStep = 0; mseF32b_reductionStep < (int)(8); ++mseF32b_reductionStep) {
    unsigned int mseF32b_stride = mseF32b_reductionStride;
    if (lid < mseF32b_stride) {
      unsigned int otherIdx2 = (lid + mseF32b_stride);
      float mine2 = sdata[lid];
      float other2 = sdata[otherIdx2];
      float combined2 = (mine2 + other2);
      sdata[lid] = combined2;
    }
    unsigned int mseF32b_strideToHalve = mseF32b_reductionStride;
    unsigned int mseF32b_nextStride = (mseF32b_strideToHalve >> 1u);
    mseF32b_reductionStride = mseF32b_nextStride;
    __syncthreads();
  }
  if (isFirst) {
    float totalDevSq = sdata[0];
    float nSamplesF2 = ((float)(nSamples));
    float mseValue = (totalDevSq / nSamplesF2);
    mse[0] = mseValue;
  }
}
