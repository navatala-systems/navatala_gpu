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
extern "C" __global__ void navatala_dataframe_mape_f32(const float* y_true, const float* y_pred, const unsigned int* n, float* result) {
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
  float absTrueVal = abs(trueVal);
  float pctErr = (absErr / absTrueVal);
  if (inBounds) {
    sdata[lid] = pctErr;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int mapeF32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int mapeF32_stride = mapeF32_reductionStride;
    if (lid < mapeF32_stride) {
      unsigned int mapeF32_partnerIdx = (lid + mapeF32_stride);
      float other = sdata[mapeF32_partnerIdx];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int mapeF32_strideToHalve = mapeF32_reductionStride;
    unsigned int mapeF32_nextStride = (mapeF32_strideToHalve >> 1u);
    mapeF32_reductionStride = mapeF32_nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if (lid == zeroU32) {
    float totalSum = sdata[0];
    float countFloat = ((float)(countVal));
    float avgPctErr = (totalSum / countFloat);
    float mapeVal = (__uint_as_float(0x42c80000u) * avgPctErr);
    result[0] = mapeVal;
  }
}
