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
__kernel void navatala_dataframe_tdigest_cdf_f64(__global const double* x, __global const double* centroidMeans, __global const double* centroidWeights, __global const uint* centroidCount, __global const double* totalWeight, __global const double* minVal, __global const double* maxVal, __global double* cdf) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if (gid == (uint)(0u)) {
    double xVal = x[0];
    uint count = centroidCount[0];
    double minV = minVal[0];
    double maxV = maxVal[0];
    double range = (maxV - minV);
    double safeRange = (((range == as_double(0x0000000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (range));
    double rawCdf = ((xVal - minV) / safeRange);
    double clampLo = (((rawCdf < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (rawCdf));
    double clampHi = (((clampLo > as_double(0x3ff0000000000000ul))) ? (as_double(0x3ff0000000000000ul)) : (clampLo));
    bool isEmpty = (count == (uint)(0u));
    double finalCdf = ((isEmpty) ? (as_double(0x0000000000000000ul)) : (clampHi));
    cdf[0] = finalCdf;
  }
}
