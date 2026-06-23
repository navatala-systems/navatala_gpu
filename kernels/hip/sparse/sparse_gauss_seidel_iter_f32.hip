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
extern "C" __global__ void navatala_sparse_sparse_gauss_seidel_iter_f32(const unsigned int* rowPtr, const unsigned int* colInd, const float* values, const float* x, const float* b, const unsigned int* n, float* x_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  unsigned int nVal = n[0u];
  if (i < nVal) {
    unsigned int rowStart = rowPtr[i];
    unsigned int iPlusOne = (i + 1u);
    unsigned int rowEnd = rowPtr[iPlusOne];
    float sumAccum = __uint_as_float(0x00000000u);
    float diagAccum = __uint_as_float(0x3f800000u);
    float bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      if (jU32 >= rowStart) {
        unsigned int col = colInd[jU32];
        float aVal = values[jU32];
        bool isDiag = (col == i);
        bool isLower = (col < i);
        float currentDiag = diagAccum;
        float newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        float xjNew = x_new[col];
        float xjOld = x[col];
        float xj = ((isLower) ? (xjNew) : (xjOld));
        float contrib = (aVal * xj);
        float offDiagContrib = ((isDiag) ? (__uint_as_float(0x00000000u)) : (contrib));
        float currentSum = sumAccum;
        float newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    float finalSum = sumAccum;
    float finalDiag = diagAccum;
    float numerator = (bi - finalSum);
    float result = (numerator / finalDiag);
    x_new[i] = result;
  }
}
