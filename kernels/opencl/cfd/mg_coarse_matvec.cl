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

__kernel void navatala_cfd_mg_coarse_matvec(__global const int* edgeU, __global const int* edgeV, __global const float* edgeCf, __global const float* diag, __global const float* x, __global float* outAx, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if (((int)(get_global_id(0)) >= total)) {
    return;
  } else {
    if (((int)(get_global_id(0)) < ((int)(mgCounts[1])))) {
      atomic_add(&outAx[(int)(get_global_id(0))], (diag[(int)(get_global_id(0))] * x[(int)(get_global_id(0))]));
    } else {
      int e = ((int)(get_global_id(0)) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomic_add(&outAx[u], (cf * x[v]));
      atomic_add(&outAx[v], (cf * x[u]));
    }
  }
}
