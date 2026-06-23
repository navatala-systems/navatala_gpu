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

__kernel void navatala_cfd_mg_coarse_jacobi(__global float* x, __global const float* r, __global const float* diag, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    float d = diag[(int)(get_global_id(0))];
    if (d == as_float(0x00000000u)) {
      return;
    } else {
      float omega = as_float(0x3f4ccccdu);
      x[(int)(get_global_id(0))] = (x[(int)(get_global_id(0))] + (omega * (r[(int)(get_global_id(0))] / d)));
    }
  }
}
