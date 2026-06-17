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
extern "C" __global__ void navatala_linalg_transpose_in_place_f64(const unsigned int* n, double* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0];
  unsigned int numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    double gidF = ((double)(gid));
    double dimF = ((double)(dim));
    double twoNm1 = ((__longlong_as_double(0x4000000000000000ull) * dimF) - __longlong_as_double(0x3ff0000000000000ull));
    double disc = ((twoNm1 * twoNm1) - (__longlong_as_double(0x4020000000000000ull) * gidF));
    double sqrtDisc = sqrt(disc);
    double iF = floor(((twoNm1 - sqrtDisc) / __longlong_as_double(0x4000000000000000ull)));
    unsigned int i = ((unsigned int)(iF));
    unsigned int rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    unsigned int offset = (gid - rowStart);
    unsigned int j = ((i + 1u) + offset);
    unsigned int idx_ij = ((i * dim) + j);
    unsigned int idx_ji = ((j * dim) + i);
    double val_ij = A[idx_ij];
    double val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}
