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
__kernel void navatala_dataframe_covariance_f64(__global const double* inputX, __global const double* inputY, __global const uint* count, __global const double* meanX, __global const double* meanY, __global double* covariance) {
  int gid0 = (int)get_global_id(0);
  __local double sdata[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  uint countVal = count[0];
  double meanXVal = meanX[0];
  double meanYVal = meanY[0];
  bool inBounds = (gid < countVal);
  double x = inputX[gid];
  double y = inputY[gid];
  double dx = (x - meanXVal);
  double dy = (y - meanYVal);
  double product = (dx * dy);
  if (inBounds) {
    sdata[lid] = product;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint covF64_reductionStride = (uint)(128u);
  for (int covF64_reductionStep = 0; covF64_reductionStep < (int)(8); ++covF64_reductionStep) {
    uint covF64_stride = covF64_reductionStride;
    if ((lid < covF64_stride)) {
      double covF64_other = sdata[(lid + covF64_stride)];
      double covF64_mine = sdata[lid];
      double covF64_sum = (covF64_mine + covF64_other);
      sdata[lid] = covF64_sum;
    }
    uint covF64_strideToHalve = covF64_reductionStride;
    uint covF64_nextStride = (covF64_strideToHalve >> (uint)(1u));
    covF64_reductionStride = covF64_nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double totalSum = sdata[(uint)(0u)];
    double countFloat = ((double)(countVal));
    double covVal = (totalSum / countFloat);
    covariance[(uint)(0u)] = covVal;
  }
}
