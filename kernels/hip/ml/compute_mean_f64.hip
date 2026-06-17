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
extern "C" __global__ void navatala_ml_compute_mean_f64(const double* y, const unsigned int* nSamples, double* mean) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int n = nSamples[0];
  bool inBounds = (gid < n);
  double val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int meanF64Stride = 128u;
  for (int reductionStep6 = 0; reductionStep6 < (int)(8); ++reductionStep6) {
    unsigned int stride6 = meanF64Stride;
    if ((lid < stride6)) {
      double other = sdata[(lid + stride6)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int strideToHalve6 = meanF64Stride;
    unsigned int nextStride6 = (strideToHalve6 >> 1u);
    meanF64Stride = nextStride6;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double totalSum = sdata[0];
    double nFloat = ((double)(n));
    double meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}
