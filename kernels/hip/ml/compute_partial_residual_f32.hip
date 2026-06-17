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
extern "C" __global__ void navatala_ml_compute_partial_residual_f32(const float* X, const float* beta, const float* y, const unsigned int* j, const unsigned int* nSamples, const unsigned int* nFeatures, float* residuals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int jVal = j[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    float yVal = y[gid];
    float betaJ = beta[jVal];
    unsigned int rowOffset = (gid * d);
    unsigned int xijIdx = (rowOffset + jVal);
    float xij = X[xijIdx];
    float xijBetaJ = (xij * betaJ);
    float dotAccum = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float diff = (yVal - dotProduct);
    float residVal = (diff + xijBetaJ);
    residuals[gid] = residVal;
  }
}
