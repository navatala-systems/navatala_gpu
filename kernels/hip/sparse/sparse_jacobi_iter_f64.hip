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
extern "C" __global__ void navatala_sparse_sparse_jacobi_iter_f64(const unsigned int* rowPtr, const unsigned int* colInd, const double* values, const double* x, const double* b, const unsigned int* n, double* x_new) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  unsigned int nVal = n[0u];
  if ((i < nVal)) {
    unsigned int rowStart = rowPtr[i];
    unsigned int iPlusOne = (i + 1u);
    unsigned int rowEnd = rowPtr[iPlusOne];
    double sumAccum = __longlong_as_double(0x0000000000000000ull);
    double diagAccum = __longlong_as_double(0x3ff0000000000000ull);
    double bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      if ((jU32 >= rowStart)) {
        unsigned int col = colInd[jU32];
        double aVal = values[jU32];
        bool isDiag = (col == i);
        double currentDiag = diagAccum;
        double newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        double xj = x[col];
        double contrib = (aVal * xj);
        double offDiagContrib = ((isDiag) ? (__longlong_as_double(0x0000000000000000ull)) : (contrib));
        double currentSum = sumAccum;
        double newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    double finalSum = sumAccum;
    double finalDiag = diagAccum;
    double numerator = (bi - finalSum);
    double result = (numerator / finalDiag);
    x_new[i] = result;
  }
}
