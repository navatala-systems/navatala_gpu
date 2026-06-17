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
extern "C" __global__ void navatala_ml_compute_m_s_e_f64(const double* targets, const unsigned int* sample_indices, const unsigned int* n_node_samples, double* mse, double* mean_val) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nSamples = n_node_samples[0];
  bool inBounds = (gid < nSamples);
  __shared__ double sdata[256];
  if (inBounds) {
    unsigned int sampleIdx = sample_indices[gid];
    double targetVal = targets[sampleIdx];
    sdata[lid] = targetVal;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int mseF64_reductionStride = 128u;
  for (int mseF64_reductionStep = 0; mseF64_reductionStep < (int)(8); ++mseF64_reductionStep) {
    unsigned int mseF64_stride = mseF64_reductionStride;
    if ((lid < mseF64_stride)) {
      unsigned int otherIdx = (lid + mseF64_stride);
      double mine = sdata[lid];
      double other = sdata[otherIdx];
      double combined = (mine + other);
      sdata[lid] = combined;
    }
    unsigned int mseF64_strideToHalve = mseF64_reductionStride;
    unsigned int mseF64_nextStride = (mseF64_strideToHalve >> 1u);
    mseF64_reductionStride = mseF64_nextStride;
    __syncthreads();
  }
  bool isFirst = (lid == 0u);
  if (isFirst) {
    double totalSum = sdata[0];
    double nSamplesF = ((double)(nSamples));
    double meanValue = (totalSum / nSamplesF);
    mean_val[0] = meanValue;
  }
  __syncthreads();
  double meanForDev = mean_val[0];
  if (inBounds) {
    unsigned int sampleIdx2 = sample_indices[gid];
    double targetVal2 = targets[sampleIdx2];
    double dev = (targetVal2 - meanForDev);
    double devSq = (dev * dev);
    sdata[lid] = devSq;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int mseF64b_reductionStride = 128u;
  for (int mseF64b_reductionStep = 0; mseF64b_reductionStep < (int)(8); ++mseF64b_reductionStep) {
    unsigned int mseF64b_stride = mseF64b_reductionStride;
    if ((lid < mseF64b_stride)) {
      unsigned int otherIdx2 = (lid + mseF64b_stride);
      double mine2 = sdata[lid];
      double other2 = sdata[otherIdx2];
      double combined2 = (mine2 + other2);
      sdata[lid] = combined2;
    }
    unsigned int mseF64b_strideToHalve = mseF64b_reductionStride;
    unsigned int mseF64b_nextStride = (mseF64b_strideToHalve >> 1u);
    mseF64b_reductionStride = mseF64b_nextStride;
    __syncthreads();
  }
  if (isFirst) {
    double totalDevSq = sdata[0];
    double nSamplesF2 = ((double)(nSamples));
    double mseValue = (totalDevSq / nSamplesF2);
    mse[0] = mseValue;
  }
}
