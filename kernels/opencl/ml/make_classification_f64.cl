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
__kernel void navatala_ml_make_classification_f64(__global const double* X, __global const double* w, __global const double* bias, __global const uint* n, __global const uint* d, __global uint* labels) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint numN = n[0];
  uint numD = d[0];
  bool inBounds = (gid < numN);
  if (inBounds) {
    double biasVal = bias[0];
    uint sampleIdx = gid;
    uint baseIdx = (sampleIdx * numD);
    double x0 = X[baseIdx];
    double w0 = w[0];
    double dot = (x0 * w0);
    double decision = (dot + biasVal);
    bool isPositive = (decision >= as_double(0x0000000000000000ul));
    uint label = ((isPositive) ? ((uint)(1u)) : ((uint)(0u)));
    labels[gid] = label;
  }
}
