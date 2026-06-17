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
extern "C" __global__ void navatala_ml_dema_update_f64(const double* x, const double* alpha, double* ema1, double* ema2, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    double xVal = x[0];
    double a = alpha[0];
    double old1 = ema1[0];
    double old2 = ema2[0];
    unsigned int n = count[0];
    double omA = (__longlong_as_double(0x3ff0000000000000ull) - a);
    double aX = (a * xVal);
    double omA1 = (omA * old1);
    double e1u = (aX + omA1);
    double new1 = (((n == 0u)) ? (xVal) : (e1u));
    double aE1 = (a * new1);
    double omA2 = (omA * old2);
    double e2u = (aE1 + omA2);
    double new2 = (((n == 0u)) ? (new1) : (e2u));
    ema1[0] = new1;
    ema2[0] = new2;
    unsigned int nc = (n + 1u);
    count[0] = nc;
  }
}
