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
extern "C" __global__ void navatala_sparse_csr_spmv_row_f32(const unsigned int* rowPtr, const unsigned int* colInd, const float* values, const float* x, const unsigned int* n, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  unsigned int nVal = n[0u];
  if ((i < nVal)) {
    unsigned int rowStart = rowPtr[i];
    unsigned int iPlusOne = (i + 1u);
    unsigned int rowEnd = rowPtr[iPlusOne];
    float sumAccum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      if ((jU32 >= rowStart)) {
        unsigned int col = colInd[jU32];
        float aVal = values[jU32];
        float xj = x[col];
        float term = (aVal * xj);
        float currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    y[i] = sumAccum;
  }
}
