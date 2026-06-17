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
extern "C" __global__ void navatala_sparse_csr_truncate_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* tol, const unsigned int* nRows, unsigned int* newRowNnz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float rowMax = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      float a = values[k];
      rowMax = (((abs(a) > rowMax)) ? (abs(a)) : (rowMax));
    }
    float tolVal = tol[0];
    float thresh = (tolVal * rowMax);
    int count = 0;
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      float a2 = values[k2];
      count = (count + (((abs(a2) >= thresh)) ? (1) : (0)));
    }
    newRowNnz[gid] = ((unsigned int)(count));
  }
}
