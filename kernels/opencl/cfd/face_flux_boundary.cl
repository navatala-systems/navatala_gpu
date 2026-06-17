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

__kernel void navatala_cfd_face_flux_boundary(__global const float* x, __global const int* owner, __global const float* cf, __global const float* bcVal, __global const uint* bcMask, __global const uint* params, __global float* faceFlux) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    int gf = (((int)(params[1])) + (int)(get_global_id(0)));
    int o = owner[gf];
    float xo = x[o];
    uint m = bcMask[gf];
    int mInt = ((int)(m));
    if ((mInt == 4)) {
      faceFlux[gf] = bcVal[gf];
    } else {
      float other = xo;
      if ((mInt == 1)) {
        other = bcVal[gf];
      } else {
        if ((mInt == 3)) {
          other = bcVal[gf];
        } else {
          if ((mInt == 2)) {
            other = as_float(0x00000000u);
          }
        }
      }
      faceFlux[gf] = (cf[gf] * (other - xo));
    }
  }
}
