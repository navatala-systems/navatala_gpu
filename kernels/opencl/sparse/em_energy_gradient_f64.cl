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
__kernel void navatala_sparse_em_energy_gradient_f64(__global const uint* ArowPtr, __global const uint* AcolIdx, __global const double* Avalues, __global const uint* ProwPtr, __global const uint* PcolIdx, __global const double* Pvalues, __global const uint* nRows, __global double* gradient) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int rs = ((int)(ArowPtr[gid]));
    int re = ((int)(ArowPtr[(gid + 1)]));
    double sum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(AcolIdx[k]));
      double a = Avalues[k];
      int pColRs = ((int)(ProwPtr[col]));
      int pColRe = ((int)(ProwPtr[(col + 1)]));
      for (int jj = 0; jj < (int)((pColRe - pColRs)); ++jj) {
        int pk = (pColRs + jj);
        double pv = Pvalues[pk];
        sum = (sum + (a * pv));
      }
    }
    gradient[gid] = sum;
  }
}
