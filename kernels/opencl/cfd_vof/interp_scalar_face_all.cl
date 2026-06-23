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

__kernel void navatala_cfd_vof_interp_scalar_face_all(__global const float* alpha, __global const int* owner, __global const int* nei, __global const float* w, __global const float* bcAlphaF, __global const uint* params, __global float* alphaF) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(get_global_id(0)) < ((int)(params[1]))) {
      int o = owner[(int)(get_global_id(0))];
      int n = nei[(int)(get_global_id(0))];
      float wf = w[(int)(get_global_id(0))];
      float wc = (wf * ((float)((wf > as_float(0x00000000u)))));
      float a = ((wc * alpha[o]) + ((as_float(0x3f800000u) - wc) * alpha[n]));
      float ac = (a * ((float)((a > as_float(0x00000000u)))));
      if (ac > as_float(0x3f800000u)) {
        alphaF[(int)(get_global_id(0))] = as_float(0x3f800000u);
      } else {
        alphaF[(int)(get_global_id(0))] = ac;
      }
    } else {
      float a = bcAlphaF[(int)(get_global_id(0))];
      float ac = (a * ((float)((a > as_float(0x00000000u)))));
      if (ac > as_float(0x3f800000u)) {
        alphaF[(int)(get_global_id(0))] = as_float(0x3f800000u);
      } else {
        alphaF[(int)(get_global_id(0))] = ac;
      }
    }
  }
}
