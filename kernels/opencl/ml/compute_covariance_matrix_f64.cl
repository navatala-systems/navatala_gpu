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
__kernel void navatala_ml_compute_covariance_matrix_f64(__global const double* Xc, __global const uint* nSamples, __global const uint* nFeatures, __global double* Cov) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint covSize = (d * d);
  bool inBounds = (gid < covSize);
  if (inBounds) {
    uint i = (gid / d);
    uint j = (gid % d);
    double sum = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(n); ++k) {
      uint kU32 = ((uint)(k));
      uint idxI = ((kU32 * d) + i);
      uint idxJ = ((kU32 * d) + j);
      double xi = Xc[idxI];
      double xj = Xc[idxJ];
      double prod = (xi * xj);
      double currSum = sum;
      double newSum = (currSum + prod);
      sum = newSum;
    }
    double finalSum = sum;
    uint nMinus1 = (n - (uint)(1u));
    double divisor = ((double)(nMinus1));
    double covVal = (finalSum / divisor);
    Cov[gid] = covVal;
  }
}
