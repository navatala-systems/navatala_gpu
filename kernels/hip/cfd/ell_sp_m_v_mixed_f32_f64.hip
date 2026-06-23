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
extern "C" __global__ void navatala_cfd_ell_sp_m_v_mixed_f32_f64(const unsigned int* colIdx, const float* values, const float* x, const unsigned int* nRows, const unsigned int* maxWidth, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (row < N) {
    int W = ((int)(maxWidth[0]));
    double acc = __longlong_as_double(0x0000000000000000ull);
    for (int col = 0; col < (int)(W); ++col) {
      int idx = (row + (col * N));
      int c = ((int)(colIdx[idx]));
      float v = values[idx];
      float xi = x[c];
      acc = (acc + (((double)(v)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}
