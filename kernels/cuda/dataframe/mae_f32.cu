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
extern "C" __global__ void navatala_dataframe_mae_f32(const float* y_true, const float* y_pred, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = n[0];
  bool inBounds = (gid < countVal);
  float trueVal = y_true[gid];
  float predVal = y_pred[gid];
  float diff = (trueVal - predVal);
  float absErr = abs(diff);
  if (inBounds) {
    sdata[lid] = absErr;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int maeF32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int maeF32_stride = maeF32_reductionStride;
    if (lid < maeF32_stride) {
      unsigned int maeF32_partnerIdx = (lid + maeF32_stride);
      float other = sdata[maeF32_partnerIdx];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int maeF32_strideToHalve = maeF32_reductionStride;
    unsigned int maeF32_nextStride = (maeF32_strideToHalve >> 1u);
    maeF32_reductionStride = maeF32_nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if (lid == zeroU32) {
    float totalSum = sdata[0];
    float countFloat = ((float)(countVal));
    float maeVal = (totalSum / countFloat);
    result[0] = maeVal;
  }
}
