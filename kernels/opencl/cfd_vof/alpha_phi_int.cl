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

__kernel void navatala_cfd_vof_alpha_phi_int(__global const float* alpha, __global const float* phi, __global const int* owner, __global const int* nei, __global const float* phir, __global const float* alphaF, __global const int* faceParams, __global float* outAlphaPhi) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float a = as_float(0x00000000u);
    if (ph >= as_float(0x00000000u)) {
      a = alpha[o];
    } else {
      a = alpha[n];
    }
    if (a < as_float(0x00000000u)) {
      a = as_float(0x00000000u);
    }
    if (a > as_float(0x3f800000u)) {
      a = as_float(0x3f800000u);
    }
    float ac = alphaF[((int)((int)(get_global_id(0))))];
    if (ac < as_float(0x00000000u)) {
      ac = as_float(0x00000000u);
    }
    if (ac > as_float(0x3f800000u)) {
      ac = as_float(0x3f800000u);
    }
    float oneMinus = (as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinus);
    outAlphaPhi[((int)((int)(get_global_id(0))))] = ((ph * a) + (phir[((int)((int)(get_global_id(0))))] * comp));
  }
}
