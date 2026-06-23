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

__kernel void navatala_cfd_primitives_phi_from_u_boundary(__global const int* owner, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* bcx, __global const float* bcy, __global const float* bcz, __global const uint* bcmask, __global const uint* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  int f = (((int)(params[1])) + (int)(get_global_id(0)));
  if (f >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[f];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    float ufx = uox;
    float ufy = uoy;
    float ufz = uoz;
    uint m = bcmask[f];
    int mInt = ((int)(m));
    if (mInt == 1) {
      float w = weights[f];
      float iw = (as_float(0x3f800000u) - w);
      ufx = ((w * uox) + (iw * bcx[f]));
      ufy = ((w * uoy) + (iw * bcy[f]));
      ufz = ((w * uoz) + (iw * bcz[f]));
    } else {
      if (mInt == 2) {
        ufx = bcx[f];
        ufy = bcy[f];
        ufz = bcz[f];
      }
    }
    outPhi[f] = (((sfX[f] * ufx) + (sfY[f] * ufy)) + (sfZ[f] * ufz));
  }
}
