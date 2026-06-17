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
extern "C" __global__ void navatala_linalg_transpose_in_place_f32(const unsigned int* n, float* A) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0];
  unsigned int numSwaps = ((dim * (dim - 1u)) / 2u);
  bool inBounds = (gid < numSwaps);
  if (inBounds) {
    float gidF = ((float)(gid));
    float dimF = ((float)(dim));
    float twoNm1 = ((__uint_as_float(0x40000000u) * dimF) - __uint_as_float(0x3f800000u));
    float disc = ((twoNm1 * twoNm1) - (__uint_as_float(0x41000000u) * gidF));
    float sqrtDisc = sqrt(disc);
    float iF = floor(((twoNm1 - sqrtDisc) / __uint_as_float(0x40000000u)));
    unsigned int i = ((unsigned int)(iF));
    unsigned int rowStart = ((i * (((2u * dim) - i) - 1u)) / 2u);
    unsigned int offset = (gid - rowStart);
    unsigned int j = ((i + 1u) + offset);
    unsigned int idx_ij = ((i * dim) + j);
    unsigned int idx_ji = ((j * dim) + i);
    float val_ij = A[idx_ij];
    float val_ji = A[idx_ji];
    A[idx_ij] = val_ji;
    A[idx_ji] = val_ij;
  }
}
