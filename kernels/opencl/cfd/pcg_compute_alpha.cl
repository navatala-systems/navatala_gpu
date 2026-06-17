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
__kernel void navatala_cfd_pcg_compute_alpha(__global const double* rz, __global const double* pAp, __global float* alpha) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= 1)) {
    return;
  } else {
    double rz_val = rz[0];
    double pAp_val = pAp[0];
    if ((pAp_val != as_double(0x0000000000000000ul))) {
      alpha[0] = ((float)((rz_val / pAp_val)));
    } else {
      alpha[0] = as_float(0x00000000u);
    }
  }
}
