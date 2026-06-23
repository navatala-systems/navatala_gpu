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
__kernel void navatala_cfd_chebyshev_smooth_post_fused_mixed(__global const float* Az, __global const float* r, __global const float* diagInv, __global const float* dPrev, __global const double* coeffA, __global const double* coeffB, __global const uint* n, __global float* z, __global float* dNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (i < N) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double az_i = ((double)(Az[i]));
    double r_i = ((double)(r[i]));
    double di_i = ((double)(diagInv[i]));
    double dp_i = ((double)(dPrev[i]));
    double z_i = ((double)(z[i]));
    double res = (r_i - az_i);
    double dinvr = (di_i * res);
    double corr = ((cA * dinvr) + (cB * dp_i));
    z[i] = ((float)((z_i + corr)));
    dNew[i] = ((float)(corr));
  }
}
