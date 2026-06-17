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
__kernel void navatala_ml_compute_correlations_f64(__global const double* X, __global const double* residual, __global const uint* nRows, __global const uint* nCols, __global double* correlations) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nRows[0];
  uint d = nCols[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint colOffset = (gid * n);
    double accum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(n); ++i) {
      uint iU32 = ((uint)(i));
      uint xIdx = (iU32 + colOffset);
      double xVal = X[xIdx];
      double rVal = residual[iU32];
      double prod = (xVal * rVal);
      double currentAccum = accum;
      double newAccum = (currentAccum + prod);
      accum = newAccum;
    }
    double corr = accum;
    correlations[gid] = corr;
  }
}
