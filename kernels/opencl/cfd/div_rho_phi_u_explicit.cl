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

__kernel void navatala_cfd_div_rho_phi_u_explicit(__global const float* phi, __global const float* rhoCell, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const float* uBcX, __global const float* uBcY, __global const float* uBcZ, __global const uint* uBcMask, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < params[1])) {
        float s = sign[k];
        float flux = phi[f];
        int o = owner[f];
        float rhoF = rhoCell[o];
        float ufx = ux[o];
        float ufy = uy[o];
        float ufz = uz[o];
        if ((f < params[2])) {
          int n = neighbour[f];
          float w0 = weights[f];
          float iw = (as_float(0x3f800000u) - w0);
          rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
          ufx = ((w0 * ux[o]) + (iw * ux[n]));
          ufy = ((w0 * uy[o]) + (iw * uy[n]));
          ufz = ((w0 * uz[o]) + (iw * uz[n]));
        } else {
          uint rm = rhoBcMask[f];
          if ((rm == (uint)(1u))) {
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[f]));
          }
          if ((rm == (uint)(2u))) {
            rhoF = rhoBcVal[f];
          }
          uint um = uBcMask[f];
          if ((um == (uint)(1u))) {
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            ufx = ((w0 * ux[o]) + (iw * uBcX[f]));
            ufy = ((w0 * uy[o]) + (iw * uBcY[f]));
            ufz = ((w0 * uz[o]) + (iw * uBcZ[f]));
          }
          if ((um == (uint)(2u))) {
            ufx = uBcX[f];
            ufy = uBcY[f];
            ufz = uBcZ[f];
          }
        }
        float rhoPhi = (rhoF * flux);
        sumX = (sumX + (s * (rhoPhi * ufx)));
        sumY = (sumY + (s * (rhoPhi * ufy)));
        sumZ = (sumZ + (s * (rhoPhi * ufz)));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if ((v != as_float(0x00000000u))) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}
