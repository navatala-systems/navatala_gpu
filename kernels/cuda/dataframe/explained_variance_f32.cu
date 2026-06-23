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
extern "C" __global__ void navatala_dataframe_explained_variance_f32(const float* y_true, const float* y_pred, const float* mean_residuals, const float* mean_true, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata_var_res[256];
  __shared__ float sdata_var_true[256];
  unsigned int countVal = n[0];
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
    sdata_var_res[lid] = __uint_as_float(0x00000000u);
    sdata_var_true[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int evF32_reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int evF32_stride = evF32_reductionStride;
    if (lid < evF32_stride) {
      unsigned int evF32_partnerIdx = (lid + evF32_stride);
      float otherRes = sdata_var_res[evF32_partnerIdx];
      float mineRes = sdata_var_res[lid];
      float sumRes = (mineRes + otherRes);
      sdata_var_res[lid] = sumRes;
      float otherTrue = sdata_var_true[evF32_partnerIdx];
      float mineTrue = sdata_var_true[lid];
      float sumTrue = (mineTrue + otherTrue);
      sdata_var_true[lid] = sumTrue;
    }
    unsigned int evF32_strideToHalve = evF32_reductionStride;
    unsigned int evF32_nextStride = (evF32_strideToHalve >> 1u);
    evF32_reductionStride = evF32_nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if (lid == zeroU32) {
    float sumSqRes = sdata_var_res[0];
    float sumSqTrue = sdata_var_true[0];
    float countFloat = ((float)(countVal));
    float varRatio = (sumSqRes / sumSqTrue);
    float evVal = (__uint_as_float(0x3f800000u) - varRatio);
    result[0] = evVal;
  }
}
