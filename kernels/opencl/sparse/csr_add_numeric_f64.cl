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
__kernel void navatala_sparse_csr_add_numeric_f64(__global const uint* ArowPtr, __global const uint* AcolIdx, __global const double* Avalues, __global const uint* BrowPtr, __global const uint* BcolIdx, __global const double* Bvalues, __global const uint* CrowPtr, __global const double* alpha, __global const double* beta, __global const uint* nRows, __global uint* CcolIdx, __global double* Cvalues) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if ((gid < N)) {
    int ars = ((int)(ArowPtr[gid]));
    int are = ((int)(ArowPtr[(gid + 1)]));
    int brs = ((int)(BrowPtr[gid]));
    int bre = ((int)(BrowPtr[(gid + 1)]));
    int crs = ((int)(CrowPtr[gid]));
    double a = alpha[0];
    double b = beta[0];
    int ci = 0;
    int ai = 0;
    int bi = 0;
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!(((ai < (are - ars)) && (bi < (bre - brs))))) break;
      int acol = ((int)(AcolIdx[(ars + ai)]));
      int bcol = ((int)(BcolIdx[(brs + bi)]));
      if ((acol == bcol)) {
        int tgt = (crs + ci);
        CcolIdx[tgt] = ((uint)(acol));
        Cvalues[tgt] = ((a * Avalues[(ars + ai)]) + (b * Bvalues[(brs + bi)]));
        ci = (ci + 1);
        ai = (ai + 1);
        bi = (bi + 1);
      } else {
        if ((acol < bcol)) {
          int tgt2 = (crs + ci);
          CcolIdx[tgt2] = ((uint)(acol));
          Cvalues[tgt2] = (a * Avalues[(ars + ai)]);
          ci = (ci + 1);
          ai = (ai + 1);
        } else {
          int tgt3 = (crs + ci);
          CcolIdx[tgt3] = ((uint)(bcol));
          Cvalues[tgt3] = (b * Bvalues[(brs + bi)]);
          ci = (ci + 1);
          bi = (bi + 1);
        }
      }
    }
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!((ai < (are - ars)))) break;
      int tgt4 = (crs + ci);
      int acol2 = ((int)(AcolIdx[(ars + ai)]));
      CcolIdx[tgt4] = ((uint)(acol2));
      Cvalues[tgt4] = (a * Avalues[(ars + ai)]);
      ci = (ci + 1);
      ai = (ai + 1);
    }
    for (int __iter = 0; __iter < 256; ++__iter) {
      if (!((bi < (bre - brs)))) break;
      int tgt5 = (crs + ci);
      int bcol2 = ((int)(BcolIdx[(brs + bi)]));
      CcolIdx[tgt5] = ((uint)(bcol2));
      Cvalues[tgt5] = (b * Bvalues[(brs + bi)]);
      ci = (ci + 1);
      bi = (bi + 1);
    }
  }
}
