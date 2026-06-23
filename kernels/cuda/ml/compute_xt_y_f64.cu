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
extern "C" __global__ void navatala_ml_compute_xt_y_f64(const double* X, const double* y, const unsigned int* nSamples, const unsigned int* nFeatures, double* Xty) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int n = nSamples[0];
  unsigned int p = nFeatures[0];
  unsigned int featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    unsigned int sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    unsigned int xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      double xVal = X[xIdx];
      double yVal = y[sampleIdx];
      double product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int xtyF64Stride = 128u;
    for (int reductionStep2 = 0; reductionStep2 < (int)(8); ++reductionStep2) {
      unsigned int stride2 = xtyF64Stride;
      if (lid < stride2) {
        double other = sdata[(lid + stride2)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int strideToHalve2 = xtyF64Stride;
      unsigned int nextStride2 = (strideToHalve2 >> 1u);
      xtyF64Stride = nextStride2;
      __syncthreads();
    }
    if (lid == 0u) {
      double totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}
