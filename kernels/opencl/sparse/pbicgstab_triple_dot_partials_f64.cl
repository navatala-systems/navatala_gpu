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
__kernel void navatala_sparse_pbicgstab_triple_dot_partials_f64(__global const double* r0hat, __global const double* r, __global const double* v, __global const double* s, __global const uint* n, __global double* partials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  double h0 = ((inBounds) ? (r0hat[gid]) : (as_double(0x0000000000000000ul)));
  double d0 = ((inBounds) ? ((h0 * r[gid])) : (as_double(0x0000000000000000ul)));
  double d1 = ((inBounds) ? ((h0 * v[gid])) : (as_double(0x0000000000000000ul)));
  double d2 = ((inBounds) ? ((h0 * s[gid])) : (as_double(0x0000000000000000ul)));
  double ws0 = sub_group_reduce_add(d0);
  double ws1 = sub_group_reduce_add(d1);
  double ws2 = sub_group_reduce_add(d2);
  __local double sd0[32];
  __local double sd1[32];
  __local double sd2[32];
  int lane = (int)(get_sub_group_local_id());
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sd0[warpIdx] = ws0;
    sd1[warpIdx] = ws1;
    sd2[warpIdx] = ws2;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8) {
    int grpId = (int)(get_group_id(0));
    int base = (grpId * 3);
    double f0 = sub_group_reduce_add(sd0[lid]);
    double f1 = sub_group_reduce_add(sd1[lid]);
    double f2 = sub_group_reduce_add(sd2[lid]);
    if (lid == 0) {
      partials[base] = f0;
      partials[(base + 1)] = f1;
      partials[(base + 2)] = f2;
    }
  }
}
