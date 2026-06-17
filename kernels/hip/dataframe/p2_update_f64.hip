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
extern "C" __global__ void navatala_dataframe_p2_update_f64(const double* x, double* q0, double* q1, double* q2, double* q3, double* q4, unsigned int* pos0, unsigned int* pos1, unsigned int* pos2, unsigned int* pos3, unsigned int* pos4, const double* targetP, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    double xVal = x[0];
    unsigned int n = count[0];
    double pVal = targetP[0];
    double h0 = q0[0];
    double h2 = q2[0];
    double h4 = q4[0];
    double newH0 = (((n == 0u)) ? (xVal) : ((((xVal < h0)) ? (xVal) : (h0))));
    double finalH0 = (((n >= 5u)) ? ((((xVal < h0)) ? (xVal) : (h0))) : (newH0));
    double finalH4 = (((n >= 5u)) ? ((((xVal > h4)) ? (xVal) : (h4))) : ((((n == 4u)) ? (xVal) : (h4))));
    double alpha = __longlong_as_double(0x3f847ae147ae147bull);
    double oneMinusAlpha = (__longlong_as_double(0x3ff0000000000000ull) - alpha);
    double finalH2 = (((n >= 5u)) ? (((oneMinusAlpha * h2) + (alpha * xVal))) : ((((n == 2u)) ? (xVal) : (h2))));
    q0[0] = finalH0;
    q2[0] = finalH2;
    q4[0] = finalH4;
    unsigned int newCount = (n + 1u);
    count[0] = newCount;
  }
}
