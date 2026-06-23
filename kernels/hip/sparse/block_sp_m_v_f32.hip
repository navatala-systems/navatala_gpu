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
extern "C" __global__ void navatala_sparse_block_sp_m_v_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* X, const unsigned int* nRows, const unsigned int* k, float* Y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = (((int)(nRows[0])) * ((int)(k[0])));
  int nR = ((int)(nRows[0]));
  if (gid < total) {
    int row = (gid % nR);
    int vecIdx = (gid / nR);
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float sum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k2 = (rs + j);
      int col = ((int)(colIdx[k2]));
      float a = values[k2];
      float x = X[((vecIdx * nR) + col)];
      sum = (sum + (a * x));
    }
    Y[gid] = sum;
  }
}
