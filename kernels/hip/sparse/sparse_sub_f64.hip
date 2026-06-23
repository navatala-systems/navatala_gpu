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
extern "C" __global__ void navatala_sparse_sparse_sub_f64(const double* valuesA, const double* valuesB, const int* nnz, double* valuesC) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (i < nnz[0u]) {
    double a = valuesA[i];
    double b = valuesB[i];
    valuesC[i] = (a - b);
  }
}
