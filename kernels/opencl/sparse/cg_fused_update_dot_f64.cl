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
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_sparse_cg_fused_update_dot_f64(__global const double* p, __global const double* Ap, __global const double* z, __global const double* alpha, __global const uint* n, __global double* x, __global double* r, __global double* rTzPartials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  double a = alpha[0];
  double pi = (((gid < N)) ? (p[gid]) : (as_double(0x0000000000000000ul)));
  double api = (((gid < N)) ? (Ap[gid]) : (as_double(0x0000000000000000ul)));
  double zi = (((gid < N)) ? (z[gid]) : (as_double(0x0000000000000000ul)));
  if (gid < N) {
    double xi = x[gid];
    double ri = r[gid];
    double xnew = (xi + (a * pi));
    double rnew = (ri - (a * api));
    x[gid] = xnew;
    r[gid] = rnew;
  }
  double rz = ((r[gid] - (a * api)) * zi);
  double warpSum = sub_group_reduce_add(rz);
  __local double sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8) {
    double val = sdata[lid];
    double finalSum = sub_group_reduce_add(val);
    if (lid == 0) {
      int grpId = (int)(get_group_id(0));
      rTzPartials[grpId] = finalSum;
    }
  }
}
