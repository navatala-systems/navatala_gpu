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
__kernel void navatala_ml_compute_mean_f64(__global const double* y, __global const uint* nSamples, __global double* mean) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint n = nSamples[0];
  bool inBounds = (gid < n);
  double val = y[gid];
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint meanF64Stride = (uint)(128u);
  for (int reductionStep6 = 0; reductionStep6 < (int)(8); ++reductionStep6) {
    uint stride6 = meanF64Stride;
    if (lid < stride6) {
      double other = sdata[(lid + stride6)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    uint strideToHalve6 = meanF64Stride;
    uint nextStride6 = (strideToHalve6 >> (uint)(1u));
    meanF64Stride = nextStride6;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double totalSum = sdata[0];
    double nFloat = ((double)(n));
    double meanVal = (totalSum / nFloat);
    mean[0] = meanVal;
  }
}
