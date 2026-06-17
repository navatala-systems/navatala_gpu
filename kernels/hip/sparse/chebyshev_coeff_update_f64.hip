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
extern "C" __global__ void navatala_sparse_chebyshev_coeff_update_f64(const double* theta, const double* delta, const double* sigma1, const double* sigmaOld, double* coeffA, double* coeffB, double* sigmaNew) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  double th = theta[0];
  double de = delta[0];
  double s1 = sigma1[0];
  double so = sigmaOld[0];
  double twoThOverDe = ((__longlong_as_double(0x4000000000000000ull) * th) / de);
  double sn = (__longlong_as_double(0x3ff0000000000000ull) / (twoThOverDe - so));
  double cA = ((__longlong_as_double(0x4000000000000000ull) * sn) / (de * s1));
  double cB = (sn * (so - __longlong_as_double(0x3ff0000000000000ull)));
  coeffA[0] = cA;
  coeffB[0] = cB;
  sigmaNew[0] = sn;
}
