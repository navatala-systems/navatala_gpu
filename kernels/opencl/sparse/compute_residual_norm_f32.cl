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

__kernel void navatala_sparse_compute_residual_norm_f32(__global const float* residual, __global const uint* n, __global float* normSq) {
  int gid0 = (int)get_global_id(0);
  __local float sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  bool inBounds = (i < nVal);
  float ri = ((inBounds) ? (residual[i]) : (as_float(0x00000000u)));
  float riSq = (ri * ri);
  sdata[lid] = riSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 0; s < (int)((uint)(128u)); ++s) {
    uint sU32 = ((uint)(s));
    if ((lid < sU32)) {
      float other = sdata[(lid + sU32)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    atomic_add(&normSq[(uint)(0u)], sdata[(uint)(0u)]);
  }
}
