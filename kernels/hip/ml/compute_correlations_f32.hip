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
extern "C" __global__ void navatala_ml_compute_correlations_f32(const float* X, const float* residual, const unsigned int* nRows, const unsigned int* nCols, float* correlations) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nRows[0];
  unsigned int d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    unsigned int colOffset = (gid * n);
    float accum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(n); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int xIdx = (iU32 + colOffset);
      float xVal = X[xIdx];
      float rVal = residual[iU32];
      float prod = (xVal * rVal);
      float currentAccum = accum;
      float newAccum = (currentAccum + prod);
      accum = newAccum;
    }
    float corr = accum;
    correlations[gid] = corr;
  }
}
