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
extern "C" __global__ void navatala_sparse_chebyshev_smooth_f64(const unsigned int* rowPtr, const unsigned int* colInd, const double* values, const double* x, const double* xPrev, const double* diagInv, const double* b, const unsigned int* n, const double* alpha, const double* beta, double* xNext) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int i = gid;
  unsigned int nVal = n[0u];
  if (i < nVal) {
    unsigned int rowStart = rowPtr[i];
    unsigned int iPlusOne = (i + 1u);
    unsigned int rowEnd = rowPtr[iPlusOne];
    double sumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int j = 0; j < (int)(rowEnd); ++j) {
      unsigned int jU32 = ((unsigned int)(j));
      if (jU32 >= rowStart) {
        unsigned int col = colInd[jU32];
        double aVal = values[jU32];
        double xj = x[col];
        double term = (aVal * xj);
        double currentSum = sumAccum;
        sumAccum = (currentSum + term);
      }
    }
    double bi = b[i];
    double xi = x[i];
    double xPrevI = xPrev[i];
    double dInv = diagInv[i];
    double alphaVal = alpha[0u];
    double betaVal = beta[0u];
    double residual = (bi - sumAccum);
    double scaledResidual = (dInv * residual);
    double history = (xi - xPrevI);
    double update = ((alphaVal * scaledResidual) + (betaVal * history));
    xNext[i] = (xi + update);
  }
}
