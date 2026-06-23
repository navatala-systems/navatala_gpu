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

__kernel void navatala_cfd_primitives_phi_from_u_internal(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* ux, __global const float* uy, __global const float* uz, __global const uint* params, __global float* outPhi) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[1]))) {
    return;
  } else {
    int o = owner[(int)(get_global_id(0))];
    int n = neighbour[(int)(get_global_id(0))];
    float w = weights[(int)(get_global_id(0))];
    float iw = (as_float(0x3f800000u) - w);
    float ufx = ((w * ux[o]) + (iw * ux[n]));
    float ufy = ((w * uy[o]) + (iw * uy[n]));
    float ufz = ((w * uz[o]) + (iw * uz[n]));
    outPhi[(int)(get_global_id(0))] = (((sfX[(int)(get_global_id(0))] * ufx) + (sfY[(int)(get_global_id(0))] * ufy)) + (sfZ[(int)(get_global_id(0))] * ufz));
  }
}
