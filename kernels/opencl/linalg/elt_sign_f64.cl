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
__kernel void navatala_linalg_elt_sign_f64(__global const double* A, __global const uint* n, __global double* C) {
  int gid0 = (int)get_global_id(0);
  uint i = ((uint)((int)(get_global_id(0))));
  if ((i < n[(uint)(0u)])) {
    double a = A[i];
    bool isNeg = (a < as_double(0x0000000000000000ul));
    double negOrZero = ((isNeg) ? (as_double(0xbff0000000000000ul)) : (as_double(0x0000000000000000ul)));
    bool isPos = (a > as_double(0x0000000000000000ul));
    double result = ((isPos) ? (as_double(0x3ff0000000000000ul)) : (negOrZero));
    C[i] = result;
  }
}
