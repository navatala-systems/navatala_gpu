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
extern "C" __global__ void navatala_cfd_layer_norm_forward(const float* x, const int* numRows, const float* gamma, const float* beta, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nRows = numRows[0];
  if ((row < nRows)) {
    int hd = 16;
    int rowBase = (row * hd);
    float sum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      sum = (sum + xv);
    }
    float hdF = ((float)(hd));
    float mean = (sum / hdF);
    float varSum = __uint_as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float diff = (xv - mean);
      varSum = (varSum + (diff * diff));
    }
    float var = (varSum / hdF);
    float invStd = sqrt((__uint_as_float(0x3f800000u) / (var + __uint_as_float(0x3727c5acu))));
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float xn = ((xv - mean) * invStd);
      float g = gamma[j];
      float b = beta[j];
      float yv = ((xn * g) + b);
      y[idx] = yv;
    }
  }
}
