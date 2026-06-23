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
extern "C" __global__ void navatala_ml_compute_k_l_divergence_f64(const double* P, const double* Q, const unsigned int* nPoints, double* kl_divergence) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    double pij = P[gid];
    double qij = Q[gid];
    double minVal = __longlong_as_double(0x3e7ad7f29abcaf48ull);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      double ratio = (pij / qij);
      double logRatio = log(ratio);
      double klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int klF64ReductionStride = 128u;
  for (int klF64ReductionStep = 0; klF64ReductionStep < (int)(8); ++klF64ReductionStep) {
    unsigned int klF64Stride = klF64ReductionStride;
    if (lid < klF64Stride) {
      double klF64Other = sdata[(lid + klF64Stride)];
      double klF64Mine = sdata[lid];
      double klF64Sum = (klF64Mine + klF64Other);
      sdata[lid] = klF64Sum;
    }
    unsigned int klF64StrideToHalve = klF64ReductionStride;
    unsigned int klF64NextStride = (klF64StrideToHalve >> 1u);
    klF64ReductionStride = klF64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    double klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}
