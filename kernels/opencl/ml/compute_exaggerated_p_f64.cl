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
__kernel void navatala_ml_compute_exaggerated_p_f64(__global double* P, __global const double* alpha, __global const uint* nPoints) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nPoints[0];
  double alphaVal = alpha[0];
  uint totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    double pVal = P[gid];
    double pExag = (pVal * alphaVal);
    P[gid] = pExag;
  }
}
