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

__kernel void navatala_cfd_primitives_interp_scalar_face(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* cellPhi, __global const float* bcValue, __global const uint* bcMask, __global const int* params, __global float* outPhiF) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float po = cellPhi[o];
    if (((int)((int)(get_global_id(0)))) < params[1]) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float pn = cellPhi[n];
      float w = weights[((int)((int)(get_global_id(0))))];
      float iw = (as_float(0x3f800000u) - w);
      outPhiF[((int)((int)(get_global_id(0))))] = ((w * po) + (iw * pn));
    } else {
      if (bcMask[((int)((int)(get_global_id(0))))] == (uint)(1u)) {
        float w = weights[((int)((int)(get_global_id(0))))];
        float iw = (as_float(0x3f800000u) - w);
        outPhiF[((int)((int)(get_global_id(0))))] = ((w * po) + (iw * bcValue[((int)((int)(get_global_id(0))))]));
      } else {
        if (bcMask[((int)((int)(get_global_id(0))))] == (uint)(2u)) {
          outPhiF[((int)((int)(get_global_id(0))))] = bcValue[((int)((int)(get_global_id(0))))];
        } else {
          outPhiF[((int)((int)(get_global_id(0))))] = po;
        }
      }
    }
  }
}
