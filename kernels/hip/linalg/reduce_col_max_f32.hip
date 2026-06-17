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
extern "C" __global__ void navatala_linalg_reduce_col_max_f32(const float* A, const unsigned int* m, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int col = gid;
  if ((col < n[0u])) {
    float initMaxVal = A[col];
    float maxAccum = initMaxVal;
    for (int i = 0; i < (int)(m[0u]); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int idx = ((iU32 * n[0u]) + col);
      float val = A[idx];
      float currentMax = maxAccum;
      float nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    float finalMax = maxAccum;
    result[col] = finalMax;
  }
}
