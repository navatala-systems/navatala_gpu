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
extern "C" __global__ void navatala_sparse_em_energy_gradient_f32(const unsigned int* ArowPtr, const unsigned int* AcolIdx, const float* Avalues, const unsigned int* ProwPtr, const unsigned int* PcolIdx, const float* Pvalues, const unsigned int* nRows, float* gradient) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(ArowPtr[gid]));
    int re = ((int)(ArowPtr[(gid + 1)]));
    float sum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(AcolIdx[k]));
      float a = Avalues[k];
      int pColRs = ((int)(ProwPtr[col]));
      int pColRe = ((int)(ProwPtr[(col + 1)]));
      for (int jj = 0; jj < (int)((pColRe - pColRs)); ++jj) {
        int pk = (pColRs + jj);
        float pv = Pvalues[pk];
        sum = (sum + (a * pv));
      }
    }
    gradient[gid] = sum;
  }
}
