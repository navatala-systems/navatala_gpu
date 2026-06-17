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
extern "C" __global__ void navatala_ml_compute_polynomial_kernel_matrix_f32(const float* X1, const float* X2, const float* gamma, const float* coef0, const int* degree, const unsigned int* n1, const unsigned int* n2, const unsigned int* n_features, float* K) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gidX = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int gidY = ((unsigned int)((int)(blockIdx.y * blockDim.y + threadIdx.y)));
  unsigned int nRows = n1[0];
  unsigned int nCols = n2[0];
  unsigned int nFeatures = n_features[0];
  float gammaVal = gamma[0];
  float coef0Val = coef0[0];
  int degreeVal = degree[0];
  bool inBoundsX = (gidX < nRows);
  bool inBoundsY = (gidY < nCols);
  bool inBounds = (inBoundsX && inBoundsY);
  if (inBounds) {
    unsigned int x1Base = (gidX * nFeatures);
    unsigned int x2Base = (gidY * nFeatures);
    float dotProduct = __uint_as_float(0x00000000u);
    for (int f = 0; f < (int)(nFeatures); ++f) {
      unsigned int fU32 = ((unsigned int)(f));
      unsigned int x1Idx = (x1Base + fU32);
      unsigned int x2Idx = (x2Base + fU32);
      float x1Val = X1[x1Idx];
      float x2Val = X2[x2Idx];
      float product = (x1Val * x2Val);
      float currDot = dotProduct;
      float newDot = (currDot + product);
      dotProduct = newDot;
    }
    float finalDot = dotProduct;
    float scaled = (gammaVal * finalDot);
    float base = (scaled + coef0Val);
    float result = __uint_as_float(0x3f800000u);
    unsigned int degreeU32 = ((unsigned int)(degreeVal));
    for (int d = 0; d < (int)(degreeU32); ++d) {
      float currResult = result;
      float newResult = (currResult * base);
      result = newResult;
    }
    float kernelVal = result;
    unsigned int outIdx = ((gidX * nCols) + gidY);
    K[outIdx] = kernelVal;
  }
}
