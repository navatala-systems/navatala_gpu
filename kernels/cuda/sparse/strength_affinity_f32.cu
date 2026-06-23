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
extern "C" __global__ void navatala_sparse_strength_affinity_f32(const unsigned int* rowPtr, const unsigned int* colIdx, const float* values, const float* theta, const unsigned int* nRows, float* strengthValues, unsigned int* strongMask) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    float diag = __uint_as_float(0x00000000u);
    for (int jd = 0; jd < (int)((re - rs)); ++jd) {
      int kd = (rs + jd);
      int colD = ((int)(colIdx[kd]));
      if (colD == gid) {
        diag = values[kd];
      }
    }
    float thetaVal = theta[0];
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      float a = values[k];
      float aff = (((col != gid)) ? ((abs(a) / abs(diag))) : (__uint_as_float(0x00000000u)));
      strengthValues[k] = aff;
      unsigned int isStrong = (((aff >= thetaVal)) ? (1u) : (0u));
      strongMask[k] = isStrong;
    }
  }
}
