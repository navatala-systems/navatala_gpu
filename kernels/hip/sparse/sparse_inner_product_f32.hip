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
extern "C" __global__ void navatala_sparse_sparse_inner_product_f32(const float* a, const float* b, const unsigned int* nnz, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lidU32 = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nnzVal = nnz[0u];
  float localSumAccum = __uint_as_float(0x00000000u);
  unsigned int idxAccum = gid;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    unsigned int currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float prod = (aVal * bVal);
      float currentSum = localSumAccum;
      float newSum = (currentSum + prod);
      localSumAccum = newSum;
    }
    unsigned int idxForUpdate = idxAccum;
    unsigned int nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalSum = localSumAccum;
  sdata[lidU32] = finalLocalSum;
  __syncthreads();
  unsigned int strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int currentStride = strideAccum;
    if (lidU32 < currentStride) {
      unsigned int otherIdx = (lidU32 + currentStride);
      float myVal = sdata[lidU32];
      float otherVal = sdata[otherIdx];
      sdata[lidU32] = (myVal + otherVal);
    }
    unsigned int strideForUpdate = strideAccum;
    unsigned int nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    __syncthreads();
  }
  if (lidU32 == 0u) {
    result[0u] = sdata[0u];
  }
}
