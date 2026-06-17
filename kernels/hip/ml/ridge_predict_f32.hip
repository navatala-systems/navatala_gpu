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
extern "C" __global__ void navatala_ml_ridge_predict_f32(const float* X, const float* beta, const unsigned int* n, const unsigned int* p, float* yPred) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int pVal = p[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      unsigned int elemIdx = ((rowIdx * pVal) + colIdx);
      float xVal = X[elemIdx];
      float betaVal = beta[colIdx];
      float product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int pred32_reductionStride = 128u;
    for (int pred32_reductionStep = 0; pred32_reductionStep < (int)(8); ++pred32_reductionStep) {
      unsigned int pred32_stride = pred32_reductionStride;
      if ((lid < pred32_stride)) {
        float pred32_other = sdata[(lid + pred32_stride)];
        float pred32_mine = sdata[lid];
        float pred32_sum = (pred32_mine + pred32_other);
        sdata[lid] = pred32_sum;
      }
      unsigned int pred32_strideToHalve = pred32_reductionStride;
      unsigned int pred32_nextStride = (pred32_strideToHalve >> 1u);
      pred32_reductionStride = pred32_nextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      float dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}
