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

__kernel void navatala_cfd_primitives_rho_phi_from_rho_and_phi(__global const float* phiAll, __global const float* rhoCell, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const int* params, __global float* outRhoPhi) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float rhoF = rhoCell[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float w0 = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w0);
      rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
    } else {
      uint m = rhoBcMask[((int)((int)(get_global_id(0))))];
      if ((m == (uint)(1u))) {
        float w0 = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w0);
        rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[((int)((int)(get_global_id(0))))]));
      }
      if ((m == (uint)(2u))) {
        rhoF = rhoBcVal[((int)((int)(get_global_id(0))))];
      }
    }
    float flux = phiAll[((int)((int)(get_global_id(0))))];
    outRhoPhi[((int)((int)(get_global_id(0))))] = (rhoF * flux);
  }
}
