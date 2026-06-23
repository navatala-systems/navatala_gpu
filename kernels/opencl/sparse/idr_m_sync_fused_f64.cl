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
__kernel void navatala_sparse_idr_m_sync_fused_f64(__global const double* P, __global const double* r, __global const double* Minv, __global const uint* n, __global const uint* s, __global double* c) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int k = (int)(get_group_id(0));
  int N = ((int)(n[0]));
  int S = ((int)(s[0]));
  double acc = as_double(0x0000000000000000ul);
  for (int j = 0; j < (int)(S); ++j) {
    double pjr = (((gid < N)) ? ((P[(gid + (j * N))] * r[gid])) : (as_double(0x0000000000000000ul)));
    double mij = Minv[(k + (j * S))];
    acc = (acc + (mij * pjr));
  }
  double warpSum = sub_group_reduce_add(acc);
  __local double sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8) {
    double sv = sdata[lid];
    double finalSum = sub_group_reduce_add(sv);
    if (lid == 0) {
      c[k] = finalSum;
    }
  }
}
