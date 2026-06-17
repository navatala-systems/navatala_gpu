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
extern "C" __global__ void navatala_dataframe_kl_divergence_f32(const float* p, const float* q, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0u];
  bool inBounds = (gid < nVal);
  float pVal = p[gid];
  float qVal = q[gid];
  bool pPositive = (pVal > __uint_as_float(0x00000000u));
  float logP = log(pVal);
  float logQ = log(qVal);
  float logRatio = (logP - logQ);
  float pLogRatio = (pVal * logRatio);
  float contrib = ((pPositive) ? (pLogRatio) : (__uint_as_float(0x00000000u)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int kl32_reductionStride = 128u;
  for (int kl32_reductionStep = 0; kl32_reductionStep < (int)(8); ++kl32_reductionStep) {
    unsigned int kl32_stride = kl32_reductionStride;
    if ((lid < kl32_stride)) {
      float kl32_other = sdata[(lid + kl32_stride)];
      float kl32_mine = sdata[lid];
      float kl32_sum = (kl32_mine + kl32_other);
      sdata[lid] = kl32_sum;
    }
    unsigned int kl32_strideToHalve = kl32_reductionStride;
    unsigned int kl32_nextStride = (kl32_strideToHalve >> 1u);
    kl32_reductionStride = kl32_nextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    result[0u] = sdata[0u];
  }
}
