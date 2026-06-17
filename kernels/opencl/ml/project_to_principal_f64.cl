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
__kernel void navatala_ml_project_to_principal_f64(__global const double* X, __global const double* V, __global const uint* nSamples, __global const uint* nFeatures, __global const uint* nComponents, __global double* Z) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  uint totalOutput = (n * kComp);
  bool inBounds = (gid < totalOutput);
  if (inBounds) {
    uint i = (gid / kComp);
    uint c = (gid % kComp);
    double sum = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)(d); ++j) {
      uint jU32 = ((uint)(j));
      uint xIdx = ((i * d) + jU32);
      uint vIdx = ((c * d) + jU32);
      double xVal = X[xIdx];
      double vVal = V[vIdx];
      double prod = (xVal * vVal);
      double currSum = sum;
      double newSum = (currSum + prod);
      sum = newSum;
    }
    double finalSum = sum;
    Z[gid] = finalSum;
  }
}
