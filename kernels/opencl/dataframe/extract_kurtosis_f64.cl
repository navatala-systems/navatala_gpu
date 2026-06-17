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
__kernel void navatala_dataframe_extract_kurtosis_f64(__global const uint* n, __global const double* M2, __global const double* M4, __global double* kurtosis) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    uint nVal = n[0];
    double m2Val = M2[0];
    double m4Val = M4[0];
    double nf = ((double)(nVal));
    double m2Sq = (m2Val * m2Val);
    double denomSafe = (((m2Val == as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (m2Sq));
    double numerator = (nf * m4Val);
    double kurtRaw = (numerator / denomSafe);
    double kurtResult = (((m2Val == as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (kurtRaw));
    kurtosis[0] = kurtResult;
  }
}
