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
extern "C" __global__ void navatala_linalg_extract_diagonal_k_f32(const float* A, const unsigned int* n, const unsigned int* k, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int size = n[0u];
  unsigned int offset = k[0u];
  unsigned int diagLen = (size - offset);
  if ((idx < diagLen)) {
    unsigned int row = idx;
    unsigned int col = (idx + offset);
    unsigned int srcIdx = ((row * size) + col);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}
