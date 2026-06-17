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
__kernel void navatala_sparse_pbicgstab_fused_update_conv_f64(__global const double* phat, __global const double* shat, __global const double* s, __global const double* t, __global const double* alpha, __global const double* omega, __global const uint* n, __global double* x, __global double* r, __global double* rrPartials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  bool inBounds = (gid < N);
  if (inBounds) {
    double a = alpha[0];
    double om = omega[0];
    double ph = phat[gid];
    double sh = shat[gid];
    double si = s[gid];
    double ti = t[gid];
    double xi = x[gid];
    double ri = (si - (om * ti));
    x[gid] = (xi + ((a * ph) + (om * sh)));
    r[gid] = ri;
  }
  double sq = ((inBounds) ? ((r[gid] * r[gid])) : (as_double(0x0000000000000000ul)));
  double warpSum = sub_group_reduce_add(sq);
  __local double sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid < 8)) {
    double val = sdata[lid];
    double finalSum = sub_group_reduce_add(val);
    if ((lid == 0)) {
      int grpId = (int)(get_group_id(0));
      rrPartials[grpId] = finalSum;
    }
  }
}
