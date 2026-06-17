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
extern "C" __global__ void navatala_linalg_transpose_batched_f64(const double* A, const unsigned int* m, const unsigned int* n, const unsigned int* batchSize, double* B) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int rows = m[0];
  unsigned int cols = n[0];
  unsigned int numBatches = batchSize[0];
  unsigned int matSize = (rows * cols);
  unsigned int totalElems = (matSize * numBatches);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int batchIdx = (gid / matSize);
    unsigned int localIdx = (gid % matSize);
    unsigned int i = (localIdx / cols);
    unsigned int j = (localIdx % cols);
    unsigned int srcIdx = ((batchIdx * matSize) + ((i * cols) + j));
    unsigned int dstIdx = ((batchIdx * matSize) + ((j * rows) + i));
    double val = A[srcIdx];
    B[dstIdx] = val;
  }
}
