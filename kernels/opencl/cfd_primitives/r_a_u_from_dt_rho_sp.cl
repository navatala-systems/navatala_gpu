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

__kernel void navatala_cfd_primitives_r_a_u_from_dt_rho_sp(__global const float* rho, __global const float* sp, __global const float* paramsF, __global const int* counts, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    float s = sp[((int)((int)(get_global_id(0))))];
    float denom = (r + (paramsF[0] * s));
    outRAU[((int)((int)(get_global_id(0))))] = (paramsF[0] / denom);
  }
}
