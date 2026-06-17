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
extern "C" __global__ void navatala_ml_pairwise_cosine_f64(const double* X, const double* Y, const unsigned int* n, const unsigned int* m, const unsigned int* d, double* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int mVal = m[0];
  unsigned int dVal = d[0];
  unsigned int totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / mVal);
    unsigned int j = (gid % mVal);
    unsigned int xBase = (i * dVal);
    unsigned int yBase = (j * dVal);
    double dotProduct = __longlong_as_double(0x0000000000000000ull);
    double normXSq = __longlong_as_double(0x0000000000000000ull);
    double normYSq = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(dVal); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int xIdx = (xBase + kU32);
      unsigned int yIdx = (yBase + kU32);
      double xVal = X[xIdx];
      double yVal = Y[yIdx];
      double prod = (xVal * yVal);
      double currentDot = dotProduct;
      double newDot = (currentDot + prod);
      dotProduct = newDot;
      double xSq = (xVal * xVal);
      double ySq = (yVal * yVal);
      double currentNormX = normXSq;
      double newNormX = (currentNormX + xSq);
      normXSq = newNormX;
      double currentNormY = normYSq;
      double newNormY = (currentNormY + ySq);
      normYSq = newNormY;
    }
    double finalDot = dotProduct;
    double finalNormXSq = normXSq;
    double finalNormYSq = normYSq;
    double normX = sqrt(finalNormXSq);
    double normY = sqrt(finalNormYSq);
    double normProduct = (normX * normY);
    bool normIsZero = (normProduct == __longlong_as_double(0x0000000000000000ull));
    if (normIsZero) {
      D[gid] = __longlong_as_double(0x0000000000000000ull);
    } else {
      double cosineSim = (finalDot / normProduct);
      double cosineDist = (__longlong_as_double(0x3ff0000000000000ull) - cosineSim);
      D[gid] = cosineDist;
    }
  }
}
