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
extern "C" __global__ void navatala_dataframe_covariance_f64(const double* inputX, const double* inputY, const unsigned int* count, const double* meanX, const double* meanY, double* covariance) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int countVal = count[0];
  double meanXVal = meanX[0];
  double meanYVal = meanY[0];
  bool inBounds = (gid < countVal);
  double x = inputX[gid];
  double y = inputY[gid];
  double dx = (x - meanXVal);
  double dy = (y - meanYVal);
  double product = (dx * dy);
  if (inBounds) {
    sdata[lid] = product;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int covF64_reductionStride = 128u;
  for (int covF64_reductionStep = 0; covF64_reductionStep < (int)(8); ++covF64_reductionStep) {
    unsigned int covF64_stride = covF64_reductionStride;
    if (lid < covF64_stride) {
      double covF64_other = sdata[(lid + covF64_stride)];
      double covF64_mine = sdata[lid];
      double covF64_sum = (covF64_mine + covF64_other);
      sdata[lid] = covF64_sum;
    }
    unsigned int covF64_strideToHalve = covF64_reductionStride;
    unsigned int covF64_nextStride = (covF64_strideToHalve >> 1u);
    covF64_reductionStride = covF64_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    double totalSum = sdata[0u];
    double countFloat = ((double)(countVal));
    double covVal = (totalSum / countFloat);
    covariance[0u] = covVal;
  }
}
