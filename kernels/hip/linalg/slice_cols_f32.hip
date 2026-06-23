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
extern "C" __global__ void navatala_linalg_slice_cols_f32(const float* A, const unsigned int* colStart, const unsigned int* colEnd, const unsigned int* numRows, const unsigned int* srcCols, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int cs = colStart[0u];
  unsigned int ce = colEnd[0u];
  unsigned int m = numRows[0u];
  unsigned int n = srcCols[0u];
  unsigned int outCols = (ce - cs);
  unsigned int totalElems = (m * outCols);
  if (idx < totalElems) {
    unsigned int outRow = (idx / outCols);
    unsigned int outCol = (idx % outCols);
    unsigned int srcCol = (cs + outCol);
    unsigned int srcIdx = ((outRow * n) + srcCol);
    float val = A[srcIdx];
    _out[idx] = val;
  }
}
