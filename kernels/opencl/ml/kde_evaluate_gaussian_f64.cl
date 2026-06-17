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
__kernel void navatala_ml_kde_evaluate_gaussian_f64(__global const double* data, __global const double* queryPoints, __global const double* bandwidth, __global const uint* n, __global const uint* m, __global double* densities) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numQueries = m[0];
  bool inBounds = (gid < numQueries);
  if (inBounds) {
    uint numSamples = n[0];
    double h = bandwidth[0];
    double x = queryPoints[gid];
    double invSqrt2Pi = as_double(0x3fd9884533d3d180ul);
    double sum = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(numSamples); ++i) {
      uint iU32 = ((uint)(i));
      double xi = data[iU32];
      double diff = (x - xi);
      double u = (diff / h);
      double uSq = (u * u);
      double negHalfUSq = (as_double(0xbfe0000000000000ul) * uSq);
      double expTerm = exp(negHalfUSq);
      double kernelVal = (invSqrt2Pi * expTerm);
      double currSum = sum;
      sum = (currSum + kernelVal);
    }
    double nFloat = ((double)(numSamples));
    double nh = (nFloat * h);
    double finalSum = sum;
    double density = (finalSum / nh);
    densities[gid] = density;
  }
}
