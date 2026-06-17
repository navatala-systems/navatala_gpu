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

__kernel void navatala_cfd_primitives_linear_upwind_scalar_face(__global const float* flux, __global const float* xCell, __global const float* gradX, __global const float* gradY, __global const float* gradZ, __global const float* limiter, __global const int* owner, __global const int* nei, __global const float* cx, __global const float* cy, __global const float* cz, __global const float* cfx, __global const float* cfy, __global const float* cfz, __global const float* bcValue, __global const uint* bcMask, __global const uint* params, __global float* outFace) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    float phif = flux[(int)(get_global_id(0))];
    int own = owner[(int)(get_global_id(0))];
    if (((int)(get_global_id(0)) < ((int)(params[1])))) {
      if ((phif >= as_float(0x00000000u))) {
        float dx = (cfx[(int)(get_global_id(0))] - cx[own]);
        float dy = (cfy[(int)(get_global_id(0))] - cy[own]);
        float dz = (cfz[(int)(get_global_id(0))] - cz[own]);
        float corr = (((dx * gradX[own]) + (dy * gradY[own])) + (dz * gradZ[own]));
        outFace[(int)(get_global_id(0))] = (xCell[own] + (limiter[own] * corr));
      } else {
        int neiCell = nei[(int)(get_global_id(0))];
        float dx = (cfx[(int)(get_global_id(0))] - cx[neiCell]);
        float dy = (cfy[(int)(get_global_id(0))] - cy[neiCell]);
        float dz = (cfz[(int)(get_global_id(0))] - cz[neiCell]);
        float corr = (((dx * gradX[neiCell]) + (dy * gradY[neiCell])) + (dz * gradZ[neiCell]));
        outFace[(int)(get_global_id(0))] = (xCell[neiCell] + (limiter[neiCell] * corr));
      }
    } else {
      if ((phif >= as_float(0x00000000u))) {
        outFace[(int)(get_global_id(0))] = xCell[own];
      } else {
        uint mask = bcMask[(int)(get_global_id(0))];
        if ((((int)(mask)) == 0)) {
          outFace[(int)(get_global_id(0))] = xCell[own];
        } else {
          outFace[(int)(get_global_id(0))] = bcValue[(int)(get_global_id(0))];
        }
      }
    }
  }
}
