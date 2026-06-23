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
__kernel void navatala_dataframe_mae_f64(__global const double* y_true, __global const double* y_pred, __global const uint* n, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint countVal = n[0];
  bool inBounds = (gid < countVal);
  double trueVal = y_true[gid];
  double predVal = y_pred[gid];
  double diff = (trueVal - predVal);
  double absErr = fabs(diff);
  if (inBounds) {
    sdata[lid] = absErr;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint maeF64_reductionStride = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint maeF64_stride = maeF64_reductionStride;
    if (lid < maeF64_stride) {
      uint maeF64_partnerIdx = (lid + maeF64_stride);
      double other = sdata[maeF64_partnerIdx];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    uint maeF64_strideToHalve = maeF64_reductionStride;
    uint maeF64_nextStride = (maeF64_strideToHalve >> (uint)(1u));
    maeF64_reductionStride = maeF64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint zeroU32 = ((uint)(0));
  if (lid == zeroU32) {
    double totalSum = sdata[0];
    double countFloat = ((double)(countVal));
    double maeVal = (totalSum / countFloat);
    result[0] = maeVal;
  }
}
