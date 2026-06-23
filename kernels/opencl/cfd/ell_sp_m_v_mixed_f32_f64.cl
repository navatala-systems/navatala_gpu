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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_ell_sp_m_v_mixed_f32_f64(__global const uint* colIdx, __global const float* values, __global const float* x, __global const uint* nRows, __global const uint* maxWidth, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int W = ((int)(maxWidth[0]));
    double acc = as_double(0x0000000000000000ul);
    for (int col = 0; col < (int)(W); ++col) {
      int idx = (row + (col * N));
      int c = ((int)(colIdx[idx]));
      float v = values[idx];
      float xi = x[c];
      acc = (acc + (((double)(v)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}
