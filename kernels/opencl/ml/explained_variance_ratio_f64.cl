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
__kernel void navatala_ml_explained_variance_ratio_f64(__global const double* eigenvalues, __global const uint* nFeatures, __global const uint* nComponents, __global double* ratios) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint d = nFeatures[0];
  uint kComp = nComponents[0];
  bool inBoundsLoad = (gid < d);
  if (inBoundsLoad) {
    double eigVal = eigenvalues[gid];
    sdata[lid] = eigVal;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint varRatioF64RedStride = (uint)(128u);
  for (int varRatioF64RedStep = 0; varRatioF64RedStep < (int)(8); ++varRatioF64RedStep) {
    uint varRatioF64Stride = varRatioF64RedStride;
    if (lid < varRatioF64Stride) {
      double other = sdata[(lid + varRatioF64Stride)];
      double mine = sdata[lid];
      double sumVal = (mine + other);
      sdata[lid] = sumVal;
    }
    uint varRatioF64StrideToHalve = varRatioF64RedStride;
    uint varRatioF64NextStride = (varRatioF64StrideToHalve >> (uint)(1u));
    varRatioF64RedStride = varRatioF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  double totalVariance = sdata[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  bool inBoundsRatio = (gid < kComp);
  if (inBoundsRatio) {
    double eigVal = eigenvalues[gid];
    double ratio = (eigVal / totalVariance);
    ratios[gid] = ratio;
  }
}
