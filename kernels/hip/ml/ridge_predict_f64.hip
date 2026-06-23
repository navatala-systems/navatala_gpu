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
extern "C" __global__ void navatala_ml_ridge_predict_f64(const double* X, const double* beta, const unsigned int* n, const unsigned int* p, double* yPred) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int pVal = p[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int colIdx = lid;
    bool colInBounds = (colIdx < pVal);
    if (colInBounds) {
      unsigned int elemIdx = ((rowIdx * pVal) + colIdx);
      double xVal = X[elemIdx];
      double betaVal = beta[colIdx];
      double product = (xVal * betaVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int pred64_reductionStride = 128u;
    for (int pred64_reductionStep = 0; pred64_reductionStep < (int)(8); ++pred64_reductionStep) {
      unsigned int pred64_stride = pred64_reductionStride;
      if (lid < pred64_stride) {
        double pred64_other = sdata[(lid + pred64_stride)];
        double pred64_mine = sdata[lid];
        double pred64_sum = (pred64_mine + pred64_other);
        sdata[lid] = pred64_sum;
      }
      unsigned int pred64_strideToHalve = pred64_reductionStride;
      unsigned int pred64_nextStride = (pred64_strideToHalve >> 1u);
      pred64_reductionStride = pred64_nextStride;
      __syncthreads();
    }
    if (lid == 0u) {
      double dotProduct = sdata[0];
      yPred[rowIdx] = dotProduct;
    }
  }
}
