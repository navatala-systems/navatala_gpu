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
__kernel void navatala_dataframe_tdigest_quantile_f64(__global const double* p, __global const double* centroidMeans, __global const double* centroidWeights, __global const uint* centroidCount, __global const double* totalWeight, __global const double* minVal, __global const double* maxVal, __global double* quantile) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if (gid == (uint)(0u)) {
    double pVal = p[0];
    uint count = centroidCount[0];
    double minV = minVal[0];
    double maxV = maxVal[0];
    double range = (maxV - minV);
    double result = (minV + (pVal * range));
    bool isEmpty = (count == (uint)(0u));
    double finalResult = ((isEmpty) ? (as_double(0x0000000000000000ul)) : (result));
    quantile[0] = finalResult;
  }
}
