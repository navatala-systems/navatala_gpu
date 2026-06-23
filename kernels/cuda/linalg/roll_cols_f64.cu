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
extern "C" __global__ void navatala_linalg_roll_cols_f64(const double* A, const unsigned int* k, const unsigned int* m, const unsigned int* n, double* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int idx = gid;
  unsigned int kVal = k[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int totalElems = (mVal * nVal);
  if (idx < totalElems) {
    unsigned int row = (idx / nVal);
    unsigned int col = (idx % nVal);
    unsigned int kMod = (kVal % nVal);
    unsigned int colPlusN = (col + nVal);
    unsigned int colMinusK = (colPlusN - kMod);
    unsigned int srcCol = (colMinusK % nVal);
    unsigned int srcIdx = ((row * nVal) + srcCol);
    double val = A[srcIdx];
    _out[idx] = val;
  }
}
