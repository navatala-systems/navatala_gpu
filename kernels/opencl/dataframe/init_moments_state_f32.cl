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

__kernel void navatala_dataframe_init_moments_state_f32(__global uint* n, __global float* M1, __global float* M2, __global float* M3, __global float* M4) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    n[0] = (uint)(0u);
    M1[0] = as_float(0x00000000u);
    M2[0] = as_float(0x00000000u);
    M3[0] = as_float(0x00000000u);
    M4[0] = as_float(0x00000000u);
  }
}
