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
extern "C" __global__ void navatala_ml_compute_xt_y_f32(const float* X, const float* y, const unsigned int* nSamples, const unsigned int* nFeatures, float* Xty) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int n = nSamples[0];
  unsigned int p = nFeatures[0];
  unsigned int featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    unsigned int sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    unsigned int xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      float xVal = X[xIdx];
      float yVal = y[sampleIdx];
      float product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int xtyF32Stride = 128u;
    for (int reductionStep1 = 0; reductionStep1 < (int)(8); ++reductionStep1) {
      unsigned int stride1 = xtyF32Stride;
      if ((lid < stride1)) {
        float other = sdata[(lid + stride1)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int strideToHalve1 = xtyF32Stride;
      unsigned int nextStride1 = (strideToHalve1 >> 1u);
      xtyF32Stride = nextStride1;
      __syncthreads();
    }
    if ((lid == 0u)) {
      float totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}
