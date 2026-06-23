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
extern "C" __global__ void navatala_dataframe_covariance_f32(const float* inputX, const float* inputY, const unsigned int* count, const float* meanX, const float* meanY, float* covariance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int countVal = count[0];
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
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int covF32_reductionStride = 128u;
  for (int covF32_reductionStep = 0; covF32_reductionStep < (int)(8); ++covF32_reductionStep) {
    unsigned int covF32_stride = covF32_reductionStride;
    if (lid < covF32_stride) {
      float covF32_other = sdata[(lid + covF32_stride)];
      float covF32_mine = sdata[lid];
      float covF32_sum = (covF32_mine + covF32_other);
      sdata[lid] = covF32_sum;
    }
    unsigned int covF32_strideToHalve = covF32_reductionStride;
    unsigned int covF32_nextStride = (covF32_strideToHalve >> 1u);
    covF32_reductionStride = covF32_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float totalSum = sdata[0u];
    float countFloat = ((float)(countVal));
    float covVal = (totalSum / countFloat);
    covariance[0u] = covVal;
  }
}
