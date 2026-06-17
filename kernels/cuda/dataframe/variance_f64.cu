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
extern "C" __global__ void navatala_dataframe_variance_f64(const double* _input, const unsigned int* count, const double* mean, double* variance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = count[0];
  double meanVal = mean[0];
  bool inBounds = (gid < countVal);
  double x = _input[gid];
  double diff = (x - meanVal);
  double diffSq = (diff * diff);
  if (inBounds) {
    sdata[lid] = diffSq;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int reductionStride = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int stride = reductionStride;
    if ((lid < stride)) {
      unsigned int partnerIdx = (lid + stride);
      double myVal = sdata[lid];
      double partnerVal = sdata[partnerIdx];
      sdata[lid] = (myVal + partnerVal);
    }
    unsigned int strideToHalve = reductionStride;
    unsigned int nextStride = (strideToHalve >> 1u);
    reductionStride = nextStride;
    __syncthreads();
  }
  unsigned int zeroU32 = ((unsigned int)(0));
  if ((lid == zeroU32)) {
    double totalSumSq = sdata[0];
    double countFloat = ((double)(countVal));
    double varianceVal = (totalSumSq / countFloat);
    variance[0] = varianceVal;
  }
}
