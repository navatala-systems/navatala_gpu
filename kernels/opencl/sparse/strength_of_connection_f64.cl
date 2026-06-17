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
__kernel void navatala_sparse_strength_of_connection_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const double* values, __global const double* theta, __global const uint* nRows, __global uint* strongMask) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int rs = ((int)(rowPtr[gid]));
    int re = ((int)(rowPtr[(gid + 1)]));
    double maxOff = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(colIdx[k]));
      if ((col != gid)) {
        double a = values[k];
        maxOff = (((fabs(a) > maxOff)) ? (fabs(a)) : (maxOff));
      }
    }
    double th = theta[0];
    double thresh = (th * maxOff);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      double a2 = values[k2];
      uint isStrong = ((((((int)(colIdx[k2])) != gid) && (fabs(a2) > thresh))) ? ((uint)(1u)) : ((uint)(0u)));
      strongMask[k2] = isStrong;
    }
  }
}
