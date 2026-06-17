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
__kernel void navatala_ml_predict_f64(__global const double* X, __global const double* beta, __global const uint* nSamples, __global const uint* nFeatures, __global double* yPred) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = nSamples[0];
  uint p = nFeatures[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * p);
    double x0 = X[baseIdx];
    double beta0 = beta[0];
    double dot = (x0 * beta0);
    yPred[gid] = dot;
  }
}
