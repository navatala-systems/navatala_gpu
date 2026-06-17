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

__kernel void navatala_cfd_turbulence_dirichlet_face_elimination(__global const int* owner, __global const int* neighbour, __global const int* constrainedMask, __global const float* constrainedValue, __global const int* nIntFaces, __global float* upper, __global float* lower, __global float* source) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= nIntFaces[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = neighbour[((int)((int)(get_global_id(0))))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if (((oC | nC) == 0)) {
      return;
    } else {
      float up = upper[((int)((int)(get_global_id(0))))];
      float lo = lower[((int)((int)(get_global_id(0))))];
      if ((oC != 0)) {
        float cv = constrainedValue[o];
        atomic_add(&source[n], (-(lo * cv)));
      }
      if ((nC != 0)) {
        float cv = constrainedValue[n];
        atomic_add(&source[o], (-(up * cv)));
      }
      upper[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      lower[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}
