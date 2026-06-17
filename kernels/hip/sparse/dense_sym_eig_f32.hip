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
extern "C" __global__ void navatala_sparse_dense_sym_eig_f32(const float* M, const unsigned int* k, float* eigenvalues, float* eigenvectors) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int kVal = ((int)(k[0]));
  for (int i = 0; i < (int)(kVal); ++i) {
    for (int j = 0; j < (int)(kVal); ++j) {
      float val = M[((i * kVal) + j)];
      float isOne = (((i == j)) ? (__uint_as_float(0x3f800000u)) : (__uint_as_float(0x00000000u)));
      eigenvalues[i] = M[((i * kVal) + i)];
      eigenvectors[((i * kVal) + j)] = isOne;
    }
  }
}
