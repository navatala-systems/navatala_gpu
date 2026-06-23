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
__kernel void navatala_cfd_grad_alpha_cell_f64(__global const double* alphaF, __global const double* sfX, __global const double* sfY, __global const double* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const double* vol, __global const int* counts, __global double* gx, __global double* gy, __global double* gz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double sx = as_double(0x0000000000000000ul);
    double sy = as_double(0x0000000000000000ul);
    double sz = as_double(0x0000000000000000ul);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double aV = ((double)(alphaF[f]));
      double s = (signV * aV);
      double sfXv = ((double)(sfX[f]));
      double sfYv = ((double)(sfY[f]));
      double sfZv = ((double)(sfZ[f]));
      sx = (sx + (s * sfXv));
      sy = (sy + (s * sfYv));
      sz = (sz + (s * sfZv));
    }
    double invV = (as_double(0x3ff0000000000000ul) / ((double)(vol[((int)((int)(get_global_id(0))))])));
    gx[((int)((int)(get_global_id(0))))] = (sx * invV);
    gy[((int)((int)(get_global_id(0))))] = (sy * invV);
    gz[((int)((int)(get_global_id(0))))] = (sz * invV);
  }
}
