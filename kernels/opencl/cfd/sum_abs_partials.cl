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

__kernel void navatala_cfd_sum_abs_partials(__global const float* a, __global const int* counts, __global float* outPartials) {
  int gid0 = (int)get_global_id(0);
  __local float tmp[256];
  float v = as_float(0x00000000u);
  if (((int)((int)(get_global_id(0)))) < counts[0]) {
    float ai = a[((int)((int)(get_global_id(0))))];
    if (ai >= as_float(0x00000000u)) {
      v = ai;
    } else {
      v = (as_float(0x00000000u) - ai);
    }
  }
  tmp[((int)((int)(get_local_id(0))))] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 128) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 128)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 64) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 64)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 32) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 32)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 16) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 16)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 8) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 8)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 4) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 4)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 2) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 2)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 1) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 1)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) == 0) {
    outPartials[((int)((int)(get_group_id(0))))] = tmp[0];
  }
}
