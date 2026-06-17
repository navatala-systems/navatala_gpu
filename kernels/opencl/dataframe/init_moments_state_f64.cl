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
__kernel void navatala_dataframe_init_moments_state_f64(__global uint* n, __global double* M1, __global double* M2, __global double* M3, __global double* M4) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) == 0)) {
    n[0] = (uint)(0u);
    M1[0] = as_double(0x0000000000000000ul);
    M2[0] = as_double(0x0000000000000000ul);
    M3[0] = as_double(0x0000000000000000ul);
    M4[0] = as_double(0x0000000000000000ul);
  }
}
