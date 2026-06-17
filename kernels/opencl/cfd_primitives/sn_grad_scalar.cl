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

__kernel void navatala_cfd_primitives_sn_grad_scalar(__global const float* cellVal, __global const int* owner, __global const int* neighbour, __global const float* delta, __global const float* faceVal, __global const uint* faceMask, __global const int* params, __global float* outSnGrad) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= params[0])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float aO = cellVal[o];
    if ((((int)((int)(get_global_id(0)))) < params[1])) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      float aN = cellVal[n];
      outSnGrad[((int)((int)(get_global_id(0))))] = (delta[((int)((int)(get_global_id(0))))] * (aN - aO));
    } else {
      if ((faceMask[((int)((int)(get_global_id(0))))] != (uint)(0u))) {
        float aN = faceVal[((int)((int)(get_global_id(0))))];
        outSnGrad[((int)((int)(get_global_id(0))))] = (delta[((int)((int)(get_global_id(0))))] * (aN - aO));
      } else {
        outSnGrad[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      }
    }
  }
}
