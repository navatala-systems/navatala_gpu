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
extern "C" __global__ void navatala_sparse_kaczmarz_sweep_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* b, const float* rowNormSq, const float* omega, const unsigned int* nRows, float* x) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int rs = ((int)(rowPtr[row]));
    int re = ((int)(rowPtr[(row + 1)]));
    float dot = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float xi = x[col];
      dot = (dot + (a * xi));
    }
    float bi = b[row];
    float rn = rowNormSq[row];
    float om = omega[0];
    float corr = (om * ((bi - dot) / rn));
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      int col2 = ((int)(colIdx[k2]));
      float a2 = values[k2];
      float _aod1 = atomicAdd(&(x[col2]), (a2 * corr));
    }
  }
}
