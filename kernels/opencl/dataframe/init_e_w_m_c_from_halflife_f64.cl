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
__kernel void navatala_dataframe_init_e_w_m_c_from_halflife_f64(__global const double* halflife, __global double* alpha, __global double* nEff, __global double* meanX, __global double* meanY, __global double* Cxy, __global double* varX, __global double* varY) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    double h = halflife[0];
    double ln2 = as_double(0x3fe62e42fefa39eful);
    double a = (ln2 / h);
    alpha[0] = a;
    nEff[0] = as_double(0x0000000000000000ul);
    meanX[0] = as_double(0x0000000000000000ul);
    meanY[0] = as_double(0x0000000000000000ul);
    Cxy[0] = as_double(0x0000000000000000ul);
    varX[0] = as_double(0x0000000000000000ul);
    varY[0] = as_double(0x0000000000000000ul);
  }
}
