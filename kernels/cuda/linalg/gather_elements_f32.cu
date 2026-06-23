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
extern "C" __global__ void navatala_linalg_gather_elements_f32(const float* A, const unsigned int* rowIdx, const unsigned int* colIdx, const unsigned int* count, const unsigned int* n, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int countVal = count[0u];
  unsigned int nVal = n[0u];
  if (tid < countVal) {
    unsigned int row = rowIdx[tid];
    unsigned int col = colIdx[tid];
    unsigned int srcIdx = ((row * nVal) + col);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}
