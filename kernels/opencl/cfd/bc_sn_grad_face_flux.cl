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

__kernel void navatala_cfd_bc_sn_grad_face_flux(__global const float* cf, __global const float* delta, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const int* counts, __global float* outFlux) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[1])) {
    return;
  } else {
    float d = delta[((int)((int)(get_global_id(0))))];
    float sg = bcSnGrad[((int)((int)(get_global_id(0))))];
    float flux = as_float(0x00000000u);
    if ((bcSnGradMask[((int)((int)(get_global_id(0))))] != 0)) {
      if ((d != as_float(0x00000000u))) {
        flux = (cf[((int)((int)(get_global_id(0))))] * (sg / d));
      }
    }
    outFlux[((int)((int)(get_global_id(0))))] = flux;
  }
}
