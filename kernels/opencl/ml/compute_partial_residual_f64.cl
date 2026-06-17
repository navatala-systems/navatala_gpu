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
__kernel void navatala_ml_compute_partial_residual_f64(__global const double* X, __global const double* beta, __global const double* y, __global const uint* j, __global const uint* nSamples, __global const uint* nFeatures, __global double* residuals) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    double yVal = y[gid];
    double betaJ = beta[jVal];
    uint rowOffset = (gid * d);
    uint xijIdx = (rowOffset + jVal);
    double xij = X[xijIdx];
    double xijBetaJ = (xij * betaJ);
    double dotAccum = as_double(0x0000000000000000ul);
    for (int k = 0; k < (int)(d); ++k) {
      uint kU32 = ((uint)(k));
      uint xikIdx = (rowOffset + kU32);
      double xik = X[xikIdx];
      double betak = beta[kU32];
      double prod = (xik * betak);
      double currentDot = dotAccum;
      double newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    double dotProduct = dotAccum;
    double diff = (yVal - dotProduct);
    double residVal = (diff + xijBetaJ);
    residuals[gid] = residVal;
  }
}
