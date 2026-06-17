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

__kernel void navatala_sparse_chebyshev_coeff_update_f32(__global const float* theta, __global const float* delta, __global const float* sigma1, __global const float* sigmaOld, __global float* coeffA, __global float* coeffB, __global float* sigmaNew) {
  int gid0 = (int)get_global_id(0);
  float th = theta[0];
  float de = delta[0];
  float s1 = sigma1[0];
  float so = sigmaOld[0];
  float twoThOverDe = ((as_float(0x40000000u) * th) / de);
  float sn = (as_float(0x3f800000u) / (twoThOverDe - so));
  float cA = ((as_float(0x40000000u) * sn) / (de * s1));
  float cB = (sn * (so - as_float(0x3f800000u)));
  coeffA[0] = cA;
  coeffB[0] = cB;
  sigmaNew[0] = sn;
}
