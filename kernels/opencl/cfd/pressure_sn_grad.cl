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

__kernel void navatala_cfd_pressure_sn_grad(__global const float* phiHbyA, __global const int* ownerAll, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* magSf, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* uBcX, __global const float* uBcY, __global const float* uBcZ, __global const uint* uBcMask, __global const float* rAUf, __global const uint* snGradMask, __global const uint* params, __global float* outSnGrad) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[1]))) {
    return;
  } else {
    uint mask = snGradMask[(int)(get_global_id(0))];
    if (((int)(mask)) == 0) {
      outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
    } else {
      int o = ownerAll[(int)(get_global_id(0))];
      if (o < 0) {
        outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          uint bcm = uBcMask[(int)(get_global_id(0))];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[(int)(get_global_id(0))];
            uyF = uBcY[(int)(get_global_id(0))];
            uzF = uBcZ[(int)(get_global_id(0))];
          }
          float phiU = (((sfX[(int)(get_global_id(0))] * uxF) + (sfY[(int)(get_global_id(0))] * uyF)) + (sfZ[(int)(get_global_id(0))] * uzF));
          float rf = rAUf[(int)(get_global_id(0))];
          float ms = magSf[(int)(get_global_id(0))];
          float denom = (rf * ms);
          float ph = phiHbyA[(int)(get_global_id(0))];
          if (fabs(denom) < as_float(0x1e3ce508u)) {
            outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
          } else {
            outSnGrad[(int)(get_global_id(0))] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}
