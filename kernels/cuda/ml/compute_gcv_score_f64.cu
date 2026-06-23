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
extern "C" __global__ void navatala_ml_compute_gcv_score_f64(const double* y, const double* yPred, const double* traceH, const unsigned int* n, double* gcv) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  double traceHVal = traceH[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    double yVal = y[gid];
    double yPredVal = yPred[gid];
    double residual = (yVal - yPredVal);
    double residualSq = (residual * residual);
    sdata[lid] = residualSq;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int gcv64_reductionStride = 128u;
  for (int gcv64_reductionStep = 0; gcv64_reductionStep < (int)(8); ++gcv64_reductionStep) {
    unsigned int gcv64_stride = gcv64_reductionStride;
    if (lid < gcv64_stride) {
      double gcv64_other = sdata[(lid + gcv64_stride)];
      double gcv64_mine = sdata[lid];
      double gcv64_sum = (gcv64_mine + gcv64_other);
      sdata[lid] = gcv64_sum;
    }
    unsigned int gcv64_strideToHalve = gcv64_reductionStride;
    unsigned int gcv64_nextStride = (gcv64_strideToHalve >> 1u);
    gcv64_reductionStride = gcv64_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    double rss = sdata[0];
    double nFloat = ((double)(nVal));
    double traceHOverN = (traceHVal / nFloat);
    double oneMinusRatio = (__longlong_as_double(0x3ff0000000000000ull) - traceHOverN);
    double denom = (oneMinusRatio * oneMinusRatio);
    double fullDenom = (nFloat * denom);
    double gcvVal = (rss / fullDenom);
    gcv[0] = gcvVal;
  }
}
