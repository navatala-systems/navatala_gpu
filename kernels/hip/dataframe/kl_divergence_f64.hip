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
extern "C" __global__ void navatala_dataframe_kl_divergence_f64(const double* p, const double* q, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0u];
  bool inBounds = (gid < nVal);
  double pVal = p[gid];
  double qVal = q[gid];
  bool pPositive = (pVal > __longlong_as_double(0x0000000000000000ull));
  double logP = log(pVal);
  double logQ = log(qVal);
  double logRatio = (logP - logQ);
  double pLogRatio = (pVal * logRatio);
  double contrib = ((pPositive) ? (pLogRatio) : (__longlong_as_double(0x0000000000000000ull)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int kl64_reductionStride = 128u;
  for (int kl64_reductionStep = 0; kl64_reductionStep < (int)(8); ++kl64_reductionStep) {
    unsigned int kl64_stride = kl64_reductionStride;
    if ((lid < kl64_stride)) {
      double kl64_other = sdata[(lid + kl64_stride)];
      double kl64_mine = sdata[lid];
      double kl64_sum = (kl64_mine + kl64_other);
      sdata[lid] = kl64_sum;
    }
    unsigned int kl64_strideToHalve = kl64_reductionStride;
    unsigned int kl64_nextStride = (kl64_strideToHalve >> 1u);
    kl64_reductionStride = kl64_nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    result[0u] = sdata[0u];
  }
}
