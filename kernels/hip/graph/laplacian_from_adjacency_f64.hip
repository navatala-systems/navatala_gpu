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
extern "C" __global__ void navatala_graph_laplacian_from_adjacency_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* n, double* laplacianValues, double* degree) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int rowIdx = gid;
  bool inBounds = (rowIdx < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[rowIdx];
    unsigned int one32 = ((unsigned int)(1));
    unsigned int rowIdxPlusOne = (rowIdx + one32);
    unsigned int rowEnd = rowPtr[rowIdxPlusOne];
    double degreeSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int absIdx = (rowStart + jU32);
      double adjVal = adjacencyValues[absIdx];
      double negAdjVal = (__longlong_as_double(0x0000000000000000ull) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      double currentDegree = degreeSumAccum;
      double newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    double finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}
