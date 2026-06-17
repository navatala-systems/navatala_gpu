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

__kernel void navatala_cfd_primitives_interp_vector_face(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* cellX, __global const float* cellY, __global const float* cellZ, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const uint* bcMask, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float uxo = cellX[o];
    float uyo = cellY[o];
    float uzo = cellZ[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float uxn = cellX[n];
      float uyn = cellY[n];
      float uzn = cellZ[n];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      outX[((int)((int)(get_global_id(0))))] = ((w * uxo) + (iw * uxn));
      outY[((int)((int)(get_global_id(0))))] = ((w * uyo) + (iw * uyn));
      outZ[((int)((int)(get_global_id(0))))] = ((w * uzo) + (iw * uzn));
    } else {
      uint m = bcMask[((int)((int)(get_global_id(0))))];
      if ((m == (uint)(1u))) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        outX[((int)((int)(get_global_id(0))))] = ((w * uxo) + (iw * bcX[((int)((int)(get_global_id(0))))]));
        outY[((int)((int)(get_global_id(0))))] = ((w * uyo) + (iw * bcY[((int)((int)(get_global_id(0))))]));
        outZ[((int)((int)(get_global_id(0))))] = ((w * uzo) + (iw * bcZ[((int)((int)(get_global_id(0))))]));
      } else {
        if ((m == (uint)(2u))) {
          outX[((int)((int)(get_global_id(0))))] = bcX[((int)((int)(get_global_id(0))))];
          outY[((int)((int)(get_global_id(0))))] = bcY[((int)((int)(get_global_id(0))))];
          outZ[((int)((int)(get_global_id(0))))] = bcZ[((int)((int)(get_global_id(0))))];
        } else {
          outX[((int)((int)(get_global_id(0))))] = uxo;
          outY[((int)((int)(get_global_id(0))))] = uyo;
          outZ[((int)((int)(get_global_id(0))))] = uzo;
        }
      }
    }
  }
}
