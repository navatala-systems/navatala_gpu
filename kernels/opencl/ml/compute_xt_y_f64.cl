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
__kernel void navatala_ml_compute_xt_y_f64(__global const double* X, __global const double* y, __global const uint* nSamples, __global const uint* nFeatures, __global double* Xty) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint wgid = ((uint)((int)(get_group_id(0))));
  __local double sdata[256];
  uint n = nSamples[0];
  uint p = nFeatures[0];
  uint featureIdx = wgid;
  bool featureInBounds = (featureIdx < p);
  if (featureInBounds) {
    uint sampleIdx = lid;
    bool sampleInBounds = (sampleIdx < n);
    uint xIdx = ((sampleIdx * p) + featureIdx);
    if (sampleInBounds) {
      double xVal = X[xIdx];
      double yVal = y[sampleIdx];
      double product = (xVal * yVal);
      sdata[lid] = product;
    } else {
      sdata[lid] = as_double(0x0000000000000000ul);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint xtyF64Stride = (uint)(128u);
    for (int reductionStep2 = 0; reductionStep2 < (int)(8); ++reductionStep2) {
      uint stride2 = xtyF64Stride;
      if ((lid < stride2)) {
        double other = sdata[(lid + stride2)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      uint strideToHalve2 = xtyF64Stride;
      uint nextStride2 = (strideToHalve2 >> (uint)(1u));
      xtyF64Stride = nextStride2;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((lid == (uint)(0u))) {
      double totalSum = sdata[0];
      Xty[featureIdx] = totalSum;
    }
  }
}
