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

__kernel void navatala_cfd_gamg_compute_scale_factor(__global const float* num, __global const float* den, __global float* sf) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= 1)) {
    return;
  } else {
    float n = num[0];
    float d = den[0];
    float dAbs = fabs(d);
    if ((dAbs > as_float(0x0da24260u))) {
      sf[0] = (n / d);
    } else {
      sf[0] = as_float(0x3f800000u);
    }
  }
}
