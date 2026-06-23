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
extern "C" __global__ void navatala_sparse_coo_transpose_f32(const unsigned int* rowInd, const unsigned int* colInd, const float* values, const unsigned int* nnz, unsigned int* rowInd_out, unsigned int* colInd_out, float* values_out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  if (i < nnz[0u]) {
    unsigned int col = colInd[i];
    rowInd_out[i] = col;
    unsigned int row = rowInd[i];
    colInd_out[i] = row;
    float val = values[i];
    values_out[i] = val;
  }
}
