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
extern "C" __global__ void navatala_linalg_diag_to_matrix_f64(const double* v, const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int totalElems = (size * size);
  if ((idx < totalElems)) {
    unsigned int row = (idx / size);
    unsigned int col = (idx % size);
    bool isDiag = (row == col);
    double diagVal = v[row];
    double result = ((isDiag) ? (diagVal) : (__longlong_as_double(0x0000000000000000ull)));
    A[idx] = result;
  }
}
