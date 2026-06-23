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
extern "C" __global__ void navatala_dataframe_cross_entropy_f32(const float* p, const float* q, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0u];
  bool inBounds = (gid < nVal);
  float pVal = p[gid];
  float qVal = q[gid];
  bool pPositive = (pVal > __uint_as_float(0x00000000u));
  float logQ = log(qVal);
  float pLogQ = (pVal * logQ);
  float contrib = ((pPositive) ? ((__uint_as_float(0xbf800000u) * pLogQ)) : (__uint_as_float(0x00000000u)));
  if (inBounds) {
    sdata[lid] = contrib;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int ce32_reductionStride = 128u;
  for (int ce32_reductionStep = 0; ce32_reductionStep < (int)(8); ++ce32_reductionStep) {
    unsigned int ce32_stride = ce32_reductionStride;
    if (lid < ce32_stride) {
      float ce32_other = sdata[(lid + ce32_stride)];
      float ce32_mine = sdata[lid];
      float ce32_sum = (ce32_mine + ce32_other);
      sdata[lid] = ce32_sum;
    }
    unsigned int ce32_strideToHalve = ce32_reductionStride;
    unsigned int ce32_nextStride = (ce32_strideToHalve >> 1u);
    ce32_reductionStride = ce32_nextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    result[0u] = sdata[0u];
  }
}
