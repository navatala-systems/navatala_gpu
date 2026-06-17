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
extern "C" __global__ void navatala_dataframe_moments_update_f64(const double* x, unsigned int* n, double* M1, double* M2, double* M3, double* M4) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    unsigned int nOld = n[0];
    double m1Old = M1[0];
    double m2Old = M2[0];
    double m3Old = M3[0];
    double m4Old = M4[0];
    double xVal = x[0];
    unsigned int nNew = (nOld + 1u);
    double nNewF = ((double)(nNew));
    double nOldF = ((double)(nOld));
    double delta = (xVal - m1Old);
    double delta_n = (delta / nNewF);
    double delta_n2 = (delta_n * delta_n);
    double term1 = ((delta * delta_n) * nOldF);
    double m1New = (m1Old + delta_n);
    double nNew2 = (nNewF * nNewF);
    double threeNNew = (__longlong_as_double(0x4008000000000000ull) * nNewF);
    double nNew2MinusThreeN = (nNew2 - threeNNew);
    double m4Coeff = (nNew2MinusThreeN + __longlong_as_double(0x4008000000000000ull));
    double m4Term1 = ((term1 * delta_n2) * m4Coeff);
    double m4Term2 = ((__longlong_as_double(0x4018000000000000ull) * delta_n2) * m2Old);
    double m4Term3 = ((__longlong_as_double(0x4010000000000000ull) * delta_n) * m3Old);
    double m4New = (((m4Old + m4Term1) + m4Term2) - m4Term3);
    double nNewMinus2 = (nNewF - __longlong_as_double(0x4000000000000000ull));
    double m3Term1 = ((term1 * delta_n) * nNewMinus2);
    double m3Term2 = ((__longlong_as_double(0x4008000000000000ull) * delta_n) * m2Old);
    double m3New = ((m3Old + m3Term1) - m3Term2);
    double m2New = (m2Old + term1);
    n[0] = nNew;
    M1[0] = m1New;
    M2[0] = m2New;
    M3[0] = m3New;
    M4[0] = m4New;
  }
}
