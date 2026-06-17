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
extern "C" __global__ void navatala_linalg_reduce_row_max_f64(const double* A, const unsigned int* m, const unsigned int* n, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int row = gid;
  if ((row < m[0u])) {
    unsigned int firstIdx = (row * n[0u]);
    double initMaxVal = A[firstIdx];
    double maxAccum = initMaxVal;
    for (int j = 0; j < (int)(n[0u]); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int idx = ((row * n[0u]) + jU32);
      double val = A[idx];
      double currentMax = maxAccum;
      double nextMax = (((val > currentMax)) ? (val) : (currentMax));
      maxAccum = nextMax;
    }
    double finalMax = maxAccum;
    result[row] = finalMax;
  }
}
