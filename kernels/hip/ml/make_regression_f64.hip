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
extern "C" __global__ void navatala_ml_make_regression_f64(const double* X, const double* w, const double* bias, const double* noise, const unsigned int* n, const unsigned int* d, double* y) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numN = n[0];
  unsigned int numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    double biasVal = bias[0];
    unsigned int sampleIdx = gid;
    unsigned int baseIdx = (sampleIdx * numD);
    double x0 = X[baseIdx];
    double w0 = w[0];
    double dot = (x0 * w0);
    double noiseVal = noise[gid];
    double withBias = (dot + biasVal);
    double result = (withBias + noiseVal);
    y[gid] = result;
  }
}
