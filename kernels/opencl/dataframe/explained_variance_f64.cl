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
__kernel void navatala_dataframe_explained_variance_f64(__global const double* y_true, __global const double* y_pred, __global const double* mean_residuals, __global const double* mean_true, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata_var_res[256];
  __local double sdata_var_true[256];
  uint countVal = n[0];
  double meanResVal = mean_residuals[0];
  double meanTrueVal = mean_true[0];
  bool inBounds = (gid < countVal);
  double trueVal = y_true[gid];
  double predVal = y_pred[gid];
  double residual = (trueVal - predVal);
  double resDiff = (residual - meanResVal);
  double sqResDiff = (resDiff * resDiff);
  double trueDiff = (trueVal - meanTrueVal);
  double sqTrueDiff = (trueDiff * trueDiff);
  if (inBounds) {
    sdata_var_res[lid] = sqResDiff;
    sdata_var_true[lid] = sqTrueDiff;
  } else {
    sdata_var_res[lid] = as_double(0x0000000000000000ul);
    sdata_var_true[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint evF64_reductionStride = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint evF64_stride = evF64_reductionStride;
    if ((lid < evF64_stride)) {
      uint evF64_partnerIdx = (lid + evF64_stride);
      double otherRes = sdata_var_res[evF64_partnerIdx];
      double mineRes = sdata_var_res[lid];
      double sumRes = (mineRes + otherRes);
      sdata_var_res[lid] = sumRes;
      double otherTrue = sdata_var_true[evF64_partnerIdx];
      double mineTrue = sdata_var_true[lid];
      double sumTrue = (mineTrue + otherTrue);
      sdata_var_true[lid] = sumTrue;
    }
    uint evF64_strideToHalve = evF64_reductionStride;
    uint evF64_nextStride = (evF64_strideToHalve >> (uint)(1u));
    evF64_reductionStride = evF64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint zeroU32 = ((uint)(0));
  if ((lid == zeroU32)) {
    double sumSqRes = sdata_var_res[0];
    double sumSqTrue = sdata_var_true[0];
    double countFloat = ((double)(countVal));
    double varRatio = (sumSqRes / sumSqTrue);
    double evVal = (as_double(0x3ff0000000000000ul) - varRatio);
    result[0] = evVal;
  }
}
