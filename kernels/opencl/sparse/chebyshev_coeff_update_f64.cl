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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_chebyshev_coeff_update_f64(__global const double* theta, __global const double* delta, __global const double* sigma1, __global const double* sigmaOld, __global double* coeffA, __global double* coeffB, __global double* sigmaNew) {
  int gid0 = (int)get_global_id(0);
  double th = theta[0];
  double de = delta[0];
  double s1 = sigma1[0];
  double so = sigmaOld[0];
  double twoThOverDe = ((as_double(0x4000000000000000ul) * th) / de);
  double sn = (as_double(0x3ff0000000000000ul) / (twoThOverDe - so));
  double cA = ((as_double(0x4000000000000000ul) * sn) / (de * s1));
  double cB = (sn * (so - as_double(0x3ff0000000000000ul)));
  coeffA[0] = cA;
  coeffB[0] = cB;
  sigmaNew[0] = sn;
}
