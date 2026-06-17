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
extern "C" __global__ void navatala_graph_modularity_matrix_vec_f64(const double* adjacencyValues, const unsigned int* rowPtr, const unsigned int* colInd, const double* x, const double* degree, const double* totalEdgeWeight, const unsigned int* n, const double* dotDX, double* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  double totalWeight = totalEdgeWeight[0];
  double dotDXVal = dotDX[0];
  bool inBounds = (gid < nVal);
  if (inBounds) {
    unsigned int rowStart = rowPtr[gid];
    unsigned int gidPlus1 = (gid + 1u);
    unsigned int rowEnd = rowPtr[gidPlus1];
    double axValAccum = __longlong_as_double(0x0000000000000000ull);
    for (int idx = 0; idx < (int)(rowEnd); ++idx) {
      unsigned int idxU32 = ((unsigned int)(idx));
      bool idxInRange = (idxU32 >= rowStart);
      if (idxInRange) {
        double aij = adjacencyValues[idxU32];
        unsigned int j = colInd[idxU32];
        double xj = x[j];
        double product = (aij * xj);
        double currentAxVal = axValAccum;
        double newAxVal = (currentAxVal + product);
        axValAccum = newAxVal;
      }
    }
    double axVal = axValAccum;
    double di = degree[gid];
    double correction = ((di * dotDXVal) / totalWeight);
    double yVal = (axVal - correction);
    y[gid] = yVal;
  }
}
