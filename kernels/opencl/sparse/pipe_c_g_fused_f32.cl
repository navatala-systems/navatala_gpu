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

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_sparse_pipe_c_g_fused_f32(__global const float* p, __global const float* Ap, __global const float* diagInv, __global const float* alpha, __global const uint* n, __global float* x, __global float* r, __global float* z, __global float* partials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  float a = alpha[0];
  bool inBounds = (gid < N);
  if (inBounds) {
    float pi = p[gid];
    float api = Ap[gid];
    float xi = x[gid];
    float ri = r[gid];
    float rnew = (ri - (a * api));
    x[gid] = (xi + (a * pi));
    r[gid] = rnew;
    float di = diagInv[gid];
    z[gid] = (di * rnew);
  }
  float sq = ((inBounds) ? ((r[gid] * r[gid])) : (as_float(0x00000000u)));
  float warpSum = sub_group_reduce_add(sq);
  __local float sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if (lane == 0) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8) {
    float val = sdata[lid];
    float finalSum = sub_group_reduce_add(val);
    if (lid == 0) {
      int grpId = (int)(get_group_id(0));
      partials[grpId] = finalSum;
    }
  }
}
