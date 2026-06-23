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
__kernel void navatala_sparse_sparse_gauss_seidel_iter_f64(__global const uint* rowPtr, __global const uint* colInd, __global const double* values, __global const double* x, __global const double* b, __global const uint* n, __global double* x_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  if (i < nVal) {
    uint rowStart = rowPtr[i];
    uint iPlusOne = (i + (uint)(1u));
    uint rowEnd = rowPtr[iPlusOne];
    double sumAccum = as_double(0x0000000000000000ul);
    double diagAccum = as_double(0x3ff0000000000000ul);
    double bi = b[i];
    for (int j = 0; j < (int)(rowEnd); ++j) {
      uint jU32 = ((uint)(j));
      if (jU32 >= rowStart) {
        uint col = colInd[jU32];
        double aVal = values[jU32];
        bool isDiag = (col == i);
        bool isLower = (col < i);
        double currentDiag = diagAccum;
        double newDiag = ((isDiag) ? (aVal) : (currentDiag));
        diagAccum = newDiag;
        double xjNew = x_new[col];
        double xjOld = x[col];
        double xj = ((isLower) ? (xjNew) : (xjOld));
        double contrib = (aVal * xj);
        double offDiagContrib = ((isDiag) ? (as_double(0x0000000000000000ul)) : (contrib));
        double currentSum = sumAccum;
        double newSum = (currentSum + offDiagContrib);
        sumAccum = newSum;
      }
    }
    double finalSum = sumAccum;
    double finalDiag = diagAccum;
    double numerator = (bi - finalSum);
    double result = (numerator / finalDiag);
    x_new[i] = result;
  }
}
