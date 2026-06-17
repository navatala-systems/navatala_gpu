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
__kernel void navatala_dataframe_r2_score_f64(__global const double* y_true, __global const double* y_pred, __global const double* y_mean, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata_res[256];
  __local double sdata_tot[256];
  uint countVal = n[0];
  double meanVal = y_mean[0];
  bool inBounds = (gid < countVal);
  double trueVal = y_true[gid];
  double predVal = y_pred[gid];
  double diffRes = (trueVal - predVal);
  double sqRes = (diffRes * diffRes);
  double diffTot = (trueVal - meanVal);
  double sqTot = (diffTot * diffTot);
  if (inBounds) {
    sdata_res[lid] = sqRes;
    sdata_tot[lid] = sqTot;
  } else {
    sdata_res[lid] = as_double(0x0000000000000000ul);
    sdata_tot[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint r2F64_reductionStride = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint r2F64_stride = r2F64_reductionStride;
    if ((lid < r2F64_stride)) {
      uint r2F64_partnerIdx = (lid + r2F64_stride);
      double otherRes = sdata_res[r2F64_partnerIdx];
      double mineRes = sdata_res[lid];
      double sumRes = (mineRes + otherRes);
      sdata_res[lid] = sumRes;
      double otherTot = sdata_tot[r2F64_partnerIdx];
      double mineTot = sdata_tot[lid];
      double sumTot = (mineTot + otherTot);
      sdata_tot[lid] = sumTot;
    }
    uint r2F64_strideToHalve = r2F64_reductionStride;
    uint r2F64_nextStride = (r2F64_strideToHalve >> (uint)(1u));
    r2F64_reductionStride = r2F64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint zeroU32 = ((uint)(0));
  if ((lid == zeroU32)) {
    double ssRes = sdata_res[0];
    double ssTot = sdata_tot[0];
    double ratio = (ssRes / ssTot);
    double r2Val = (as_double(0x3ff0000000000000ul) - ratio);
    result[0] = r2Val;
  }
}
