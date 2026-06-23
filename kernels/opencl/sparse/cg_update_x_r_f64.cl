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
__kernel void navatala_sparse_cg_update_x_r_f64(__global const double* p, __global const double* Ap, __global const double* alpha, __global const uint* n, __global double* x, __global double* r) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (gid < N) {
    double a = alpha[0];
    double pi = p[gid];
    double api = Ap[gid];
    double xi = x[gid];
    double ri = r[gid];
    x[gid] = (xi + (a * pi));
    r[gid] = (ri - (a * api));
  }
}
