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
extern "C" __global__ void navatala_linalg_slice_rows_f64(const double* A, const unsigned int* rowStart, const unsigned int* rowEnd, const unsigned int* numCols, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int rs = rowStart[0u];
  unsigned int re = rowEnd[0u];
  unsigned int n = numCols[0u];
  unsigned int numRows = (re - rs);
  unsigned int totalElems = (numRows * n);
  if ((idx < totalElems)) {
    unsigned int outRow = (idx / n);
    unsigned int outCol = (idx % n);
    unsigned int srcRow = (rs + outRow);
    unsigned int srcIdx = ((srcRow * n) + outCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}
