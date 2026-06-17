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
extern "C" __global__ void navatala_graph_modularity_matrix_vec_f32(const float* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const float* x, const float* degree, const float* totalEdgeWeight, const unsigned int* n, const float* dotDX, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  float totalWeight = totalEdgeWeight[0];
  float dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    float axValAccum = __uint_as_float(0x00000000u);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        float aij = adjacencyValues[idxU32];
        unsigned int j = colInd[idxU32];
        float xj = x[j];
        float product = (aij * xj);
        float currentAxVal = axValAccum;
        float newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    float axVal = axValAccum;
    float di = degree[gid];
    float correction = ((di * dotDXVal) / totalWeight);
    float yVal = (axVal - correction);
    y[gid] = yVal;
  }
}
