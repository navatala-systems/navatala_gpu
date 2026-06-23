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

__kernel void navatala_dataframe_init_p2_state_f32(__global const float* p, __global float* q0, __global float* q1, __global float* q2, __global float* q3, __global float* q4, __global uint* pos0, __global uint* pos1, __global uint* pos2, __global uint* pos3, __global uint* pos4, __global float* targetP, __global uint* count) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) == 0) {
    float pVal = p[0];
    q0[0] = as_float(0x00000000u);
    q1[0] = as_float(0x00000000u);
    q2[0] = as_float(0x00000000u);
    q3[0] = as_float(0x00000000u);
    q4[0] = as_float(0x00000000u);
    pos0[0] = (uint)(0u);
    pos1[0] = (uint)(0u);
    pos2[0] = (uint)(0u);
    pos3[0] = (uint)(0u);
    pos4[0] = (uint)(0u);
    targetP[0] = pVal;
    count[0] = (uint)(0u);
  }
}
