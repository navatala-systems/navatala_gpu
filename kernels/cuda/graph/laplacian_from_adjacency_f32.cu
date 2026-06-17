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
extern "C" __global__ void navatala_graph_laplacian_from_adjacency_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const unsigned int* n, float* laplacianValues, float* degree) {
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
    float degreeSumAccum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)((rowEnd - rowStart)); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      unsigned int absIdx = (rowStart + jU32);
      float adjVal = adjacencyValues[absIdx];
      float negAdjVal = (__uint_as_float(0x00000000u) - adjVal);
      laplacianValues[absIdx] = negAdjVal;
      float currentDegree = degreeSumAccum;
      float newDegree = (currentDegree + adjVal);
      degreeSumAccum = newDegree;
    }
    float finalDegree = degreeSumAccum;
    degree[rowIdx] = finalDegree;
  }
}
