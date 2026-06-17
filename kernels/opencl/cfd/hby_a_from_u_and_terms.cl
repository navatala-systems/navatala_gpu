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

__kernel void navatala_cfd_hby_a_from_u_and_terms(__global const float* ux, __global const float* uy, __global const float* uz, __global const float* lapx, __global const float* lapy, __global const float* lapz, __global const float* divx, __global const float* divy, __global const float* divz, __global const float* extrax, __global const float* extray, __global const float* extraz, __global const float* rau, __global const int* counts, __global const int* modeBuf, __global float* outx, __global float* outy, __global float* outz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float tx = lapx[((int)((int)(get_global_id(0))))];
    float ty = lapy[((int)((int)(get_global_id(0))))];
    float tz = lapz[((int)((int)(get_global_id(0))))];
    if (((modeBuf[0] == 1) || (modeBuf[0] == 3))) {
      tx = (tx - divx[((int)((int)(get_global_id(0))))]);
      ty = (ty - divy[((int)((int)(get_global_id(0))))]);
      tz = (tz - divz[((int)((int)(get_global_id(0))))]);
    }
    if (((modeBuf[0] == 2) || (modeBuf[0] == 3))) {
      tx = (tx + extrax[((int)((int)(get_global_id(0))))]);
      ty = (ty + extray[((int)((int)(get_global_id(0))))]);
      tz = (tz + extraz[((int)((int)(get_global_id(0))))]);
    }
    float a = rau[((int)((int)(get_global_id(0))))];
    outx[((int)((int)(get_global_id(0))))] = (ux[((int)((int)(get_global_id(0))))] + (a * tx));
    outy[((int)((int)(get_global_id(0))))] = (uy[((int)((int)(get_global_id(0))))] + (a * ty));
    outz[((int)((int)(get_global_id(0))))] = (uz[((int)((int)(get_global_id(0))))] + (a * tz));
  }
}
