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
__kernel void navatala_sparse_truncate_interpolation_count_f64(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const double* Pvalues, __global const double* threshold, __global const uint* nRows, __global uint* nnzPerRowOut) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double maxAbs = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      double v = Pvalues[k];
      maxAbs = (((fabs(v) > maxAbs)) ? (fabs(v)) : (maxAbs));
    }
    double thresh = (maxAbs * threshold[0]);
    uint nnz = (uint)(0u);
    for (int j2 = 0; j2 < (int)((re - rs)); ++j2) {
      int k2 = (rs + j2);
      double v2 = Pvalues[k2];
      if ((fabs(v2) >= thresh)) {
        nnz = (nnz + (uint)(1u));
      }
    }
    nnzPerRowOut[row] = nnz;
  }
}
