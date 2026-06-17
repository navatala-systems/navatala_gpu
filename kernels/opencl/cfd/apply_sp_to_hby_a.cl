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

__kernel void navatala_cfd_apply_sp_to_hby_a(__global float* hx, __global float* hy, __global float* hz, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* rAU, __global const float* sp, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float fac = (sp[((int)((int)(get_global_id(0))))] * rAU[((int)((int)(get_global_id(0))))]);
    hx[((int)((int)(get_global_id(0))))] = (hx[((int)((int)(get_global_id(0))))] - (fac * ux[((int)((int)(get_global_id(0))))]));
    hy[((int)((int)(get_global_id(0))))] = (hy[((int)((int)(get_global_id(0))))] - (fac * uy[((int)((int)(get_global_id(0))))]));
    hz[((int)((int)(get_global_id(0))))] = (hz[((int)((int)(get_global_id(0))))] - (fac * uz[((int)((int)(get_global_id(0))))]));
  }
}
