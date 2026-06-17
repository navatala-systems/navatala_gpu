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
extern "C" __global__ void navatala_sparse_small_dense_q_r_f32(const unsigned int* m, const unsigned int* nCol, float* A, float* tau) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int mVal = ((int)(m[0]));
  int nVal = ((int)(nCol[0]));
  for (int k = 0; k < (int)(nVal); ++k) {
    float norm = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)((mVal - k)); ++i) {
      float aij = A[((k * mVal) + (k + i))];
      norm = (norm + (aij * aij));
    }
    float diag = A[((k * mVal) + k)];
    float sigma = (((diag >= __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0xbf800000u)));
    float alpha = (sigma * sqrt(norm));
    tau[k] = (__uint_as_float(0x40000000u) / (alpha * (alpha + diag)));
    A[((k * mVal) + k)] = (diag + alpha);
  }
}
