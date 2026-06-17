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
extern "C" __global__ void navatala_ml_dema_update_f32(const float* x, const float* alpha, float* ema1, float* ema2, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    float xVal = x[0];
    float a = alpha[0];
    float old1 = ema1[0];
    float old2 = ema2[0];
    unsigned int n = count[0];
    float omA = (__uint_as_float(0x3f800000u) - a);
    float aX = (a * xVal);
    float omA1 = (omA * old1);
    float e1u = (aX + omA1);
    float new1 = (((n == 0u)) ? (xVal) : (e1u));
    float aE1 = (a * new1);
    float omA2 = (omA * old2);
    float e2u = (aE1 + omA2);
    float new2 = (((n == 0u)) ? (new1) : (e2u));
    ema1[0] = new1;
    ema2[0] = new2;
    unsigned int nc = (n + 1u);
    count[0] = nc;
  }
}
