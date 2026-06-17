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
extern "C" __global__ void navatala_ml_compute_r_squared_f64(const double* y, const double* yPred, const double* yMean, const unsigned int* nSamples, double* rSquared) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdataRes[256];
  __shared__ double sdataTot[256];
  unsigned int n = nSamples[0];
  double meanVal = yMean[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    double yVal = y[gid];
    double yPredVal = yPred[gid];
    double resid = (yVal - yPredVal);
    double residSq = (resid * resid);
    double dev = (yVal - meanVal);
    double devSq = (dev * dev);
    sdataRes[lid] = residSq;
    sdataTot[lid] = devSq;
  } else {
    sdataRes[lid] = __longlong_as_double(0x0000000000000000ull);
    sdataTot[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int r2F64Stride = 128u;
  for (int reductionStep4 = 0; reductionStep4 < (int)(8); ++reductionStep4) {
    unsigned int stride4 = r2F64Stride;
    if ((lid < stride4)) {
      double otherRes = sdataRes[(lid + stride4)];
      double mineRes = sdataRes[lid];
      double sumRes = (mineRes + otherRes);
      sdataRes[lid] = sumRes;
      double otherTot = sdataTot[(lid + stride4)];
      double mineTot = sdataTot[lid];
      double sumTot = (mineTot + otherTot);
      sdataTot[lid] = sumTot;
    }
    unsigned int strideToHalve4 = r2F64Stride;
    unsigned int nextStride4 = (strideToHalve4 >> 1u);
    r2F64Stride = nextStride4;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double ssRes = sdataRes[0];
    double ssTot = sdataTot[0];
    double ratio = (ssRes / ssTot);
    double r2 = (__longlong_as_double(0x3ff0000000000000ull) - ratio);
    rSquared[0] = r2;
  }
}
