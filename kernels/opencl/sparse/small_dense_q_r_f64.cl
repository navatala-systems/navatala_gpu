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
__kernel void navatala_sparse_small_dense_q_r_f64(__global const uint* m, __global const uint* nCol, __global double* A, __global double* tau) {
  int gid0 = (int)get_global_id(0);
  int mVal = ((int)(m[0]));
  int nVal = ((int)(nCol[0]));
  for (int k = 0; k < (int)(nVal); ++k) {
    double norm = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)((mVal - k)); ++i) {
      double aij = A[((k * mVal) + (k + i))];
      norm = (norm + (aij * aij));
    }
    double diag = A[((k * mVal) + k)];
    double sigma = (((diag >= as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (as_double(0xbff0000000000000ul)));
    double alpha = (sigma * sqrt(norm));
    tau[k] = (as_double(0x4000000000000000ul) / (alpha * (alpha + diag)));
    A[((k * mVal) + k)] = (diag + alpha);
  }
}
