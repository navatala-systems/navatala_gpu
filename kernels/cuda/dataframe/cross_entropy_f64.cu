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
extern "C" __global__ void navatala_dataframe_cross_entropy_f64(const double* p, const double* q, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0u];
  bool inBounds = (gid < nVal);
  double pVal = p[gid];
  double qVal = q[gid];
  bool pPositive = (pVal > __longlong_as_double(0x0000000000000000ull));
  double logQ = log(qVal);
  double pLogQ = (pVal * logQ);
  double contrib = ((pPositive) ? ((__longlong_as_double(0xbff0000000000000ull) * pLogQ)) : (__longlong_as_double(0x0000000000000000ull)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int ce64_reductionStride = 128u;
  for (int ce64_reductionStep = 0; ce64_reductionStep < (int)(8); ++ce64_reductionStep) {
    unsigned int ce64_stride = ce64_reductionStride;
    if (lid < ce64_stride) {
      double ce64_other = sdata[(lid + ce64_stride)];
      double ce64_mine = sdata[lid];
      double ce64_sum = (ce64_mine + ce64_other);
      sdata[lid] = ce64_sum;
    }
    unsigned int ce64_strideToHalve = ce64_reductionStride;
    unsigned int ce64_nextStride = (ce64_strideToHalve >> 1u);
    ce64_reductionStride = ce64_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0u] = sdata[0u];
  }
}
