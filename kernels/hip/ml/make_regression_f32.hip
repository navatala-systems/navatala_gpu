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
extern "C" __global__ void navatala_ml_make_regression_f32(const float* X, const float* w, const float* bias, const float* noise, const unsigned int* n, const unsigned int* d, float* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    float biasVal = bias[0];
    unsigned int sampleIdx = gid;
    unsigned int baseIdx = (sampleIdx * numD);
    float x0 = X[baseIdx];
    float w0 = w[0];
    float dot = (x0 * w0);
    float noiseVal = noise[gid];
    float withBias = (dot + biasVal);
    float result = (withBias + noiseVal);
    y[gid] = result;
  }
}
