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
__kernel void navatala_ml_update_coefs_f64(__global const double* equiangular, __global const double* gamma, __global const uint* activeMask, __global const uint* nCols, __global double* beta) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint d = nCols[0];
  double gammaVal = gamma[0];
  bool inBounds = (gid < d);
  if (inBounds) {
    uint mask = activeMask[gid];
    bool isActive = (mask == (uint)(1u));
    if (isActive) {
      double betaJ = beta[gid];
      double eqJ = equiangular[gid];
      double update = (gammaVal * eqJ);
      double newBeta = (betaJ + update);
      beta[gid] = newBeta;
    }
  }
}
