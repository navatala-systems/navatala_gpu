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
extern "C" __global__ void navatala_cfd_laplacian_coeffs(const float* gamma, const int* owner, const int* neighbour, const float* weights, const float* deltaCoeffs, const float* magSf, const int* counts, float* outUpper, float* outLower) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float gammaF = gamma[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float coeff = (deltaCoeffs[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * (gammaF * magSf[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]));
    outUpper[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = coeff;
    outLower[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = coeff;
  }
}
