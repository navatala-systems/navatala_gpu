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
__kernel void navatala_ml_pairwise_chebyshev_f64(__global const double* X, __global const double* Y, __global const uint* n, __global const uint* m, __global const uint* d, __global double* D) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[0];
  uint mVal = m[0];
  uint dVal = d[0];
  uint totalElems = (nVal * mVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / mVal);
    uint j = (gid % mVal);
    uint xBase = (i * dVal);
    uint yBase = (j * dVal);
    double maxDiff = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(dVal); ++k) {
      uint kU32 = ((uint)(k));
      uint xIdx = (xBase + kU32);
      uint yIdx = (yBase + kU32);
      double xVal = X[xIdx];
      double yVal = Y[yIdx];
      double diff = (xVal - yVal);
      double absDiff = fabs(diff);
      double currentMax = maxDiff;
      bool isGreater = (absDiff > currentMax);
      double newMax = ((isGreater) ? (absDiff) : (currentMax));
      maxDiff = newMax;
    }
    double finalMax = maxDiff;
    D[gid] = finalMax;
  }
}
