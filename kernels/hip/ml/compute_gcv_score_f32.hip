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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_compute_gcv_score_f32(const float* y, const float* yPred, const float* traceH, const unsigned int* n, float* gcv) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  float traceHVal = traceH[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    float yVal = y[gid];
    float yPredVal = yPred[gid];
    float residual = (yVal - yPredVal);
    float residualSq = (residual * residual);
    sdata[lid] = residualSq;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int gcv32_reductionStride = 128u;
  for (int gcv32_reductionStep = 0; gcv32_reductionStep < (int)(8); ++gcv32_reductionStep) {
    unsigned int gcv32_stride = gcv32_reductionStride;
    if (lid < gcv32_stride) {
      float gcv32_other = sdata[(lid + gcv32_stride)];
      float gcv32_mine = sdata[lid];
      float gcv32_sum = (gcv32_mine + gcv32_other);
      sdata[lid] = gcv32_sum;
    }
    unsigned int gcv32_strideToHalve = gcv32_reductionStride;
    unsigned int gcv32_nextStride = (gcv32_strideToHalve >> 1u);
    gcv32_reductionStride = gcv32_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float rss = sdata[0];
    float nFloat = ((float)(nVal));
    float traceHOverN = (traceHVal / nFloat);
    float oneMinusRatio = (__uint_as_float(0x3f800000u) - traceHOverN);
    float denom = (oneMinusRatio * oneMinusRatio);
    float fullDenom = (nFloat * denom);
    float gcvVal = (rss / fullDenom);
    gcv[0] = gcvVal;
  }
}
