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
extern "C" __global__ void navatala_sparse_ldu_to_csr_count_nnz_f32(const unsigned int* owner, const unsigned int* neighbour, const unsigned int* nInternalFaces, const unsigned int* nCells, unsigned int* rowNnz) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int face = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nFaces = ((int)(nInternalFaces[0]));
  if (face < nFaces) {
    int own = ((int)(owner[face]));
    int nbr = ((int)(neighbour[face]));
    unsigned int _a = atomicAdd(&(rowNnz[own]), 1u);
    unsigned int _b = atomicAdd(&(rowNnz[nbr]), 1u);
  }
}
