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

__kernel void navatala_cfd_primitives_phi_from_u(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float uox = ux[o];
    float uoy = uy[o];
    float uoz = uz[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      float ufx = ((w * uox) + (iw * ux[n]));
      float ufy = ((w * uoy) + (iw * uy[n]));
      float ufz = ((w * uoz) + (iw * uz[n]));
      outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
    } else {
      if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(1u))) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        float ufx = ((w * uox) + (iw * bcX[((int)((int)(get_global_id(0))))]));
        float ufy = ((w * uoy) + (iw * bcY[((int)((int)(get_global_id(0))))]));
        float ufz = ((w * uoz) + (iw * bcZ[((int)((int)(get_global_id(0))))]));
        outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
      } else {
        if ((bcMask[((int)((int)(get_global_id(0))))] == (uint)(2u))) {
          float ufx = bcX[((int)((int)(get_global_id(0))))];
          float ufy = bcY[((int)((int)(get_global_id(0))))];
          float ufz = bcZ[((int)((int)(get_global_id(0))))];
          outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * ufx) + (sfY[((int)((int)(get_global_id(0))))] * ufy)) + (sfZ[((int)((int)(get_global_id(0))))] * ufz));
        } else {
          outPhi[((int)((int)(get_global_id(0))))] = (((sfX[((int)((int)(get_global_id(0))))] * uox) + (sfY[((int)((int)(get_global_id(0))))] * uoy)) + (sfZ[((int)((int)(get_global_id(0))))] * uoz));
        }
      }
    }
  }
}
