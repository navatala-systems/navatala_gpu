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
#include <hip/hip_bfloat16.h>
extern "C" __global__ void navatala_linalg_reduce_sum_bf16(const __hip_bfloat16* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    __hip_bfloat16 xbf16 = _input[gid];
    float xf32 = ((float)(xbf16));
    sdata[lid] = xf32;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int sumBF16ReductionStride = 128u;
  for (int sumBF16ReductionStep = 0; sumBF16ReductionStep < (int)(8); ++sumBF16ReductionStep) {
    unsigned int sumBF16Stride = sumBF16ReductionStride;
    if (lid < sumBF16Stride) {
      float other = sdata[(lid + sumBF16Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int sumBF16StrideToHalve = sumBF16ReductionStride;
    unsigned int sumBF16NextStride = (sumBF16StrideToHalve >> 1u);
    sumBF16ReductionStride = sumBF16NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    result[0] = partialSum;
  }
}
