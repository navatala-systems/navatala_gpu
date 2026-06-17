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
extern "C" __global__ void navatala_ml_compute_mean_f32(const float* y, const unsigned int* nSamples, float* mean) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int n = nSamples[0];
  bool inBounds = (gid < n);
  float val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int meanF32Stride = 128u;
  for (int reductionStep5 = 0; reductionStep5 < (int)(8); ++reductionStep5) {
    unsigned int stride5 = meanF32Stride;
    if ((lid < stride5)) {
      float other = sdata[(lid + stride5)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int strideToHalve5 = meanF32Stride;
    unsigned int nextStride5 = (strideToHalve5 >> 1u);
    meanF32Stride = nextStride5;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float totalSum = sdata[0];
    float nFloat = ((float)(n));
    float meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}
