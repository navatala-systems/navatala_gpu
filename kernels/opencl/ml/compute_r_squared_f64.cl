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
__kernel void navatala_ml_compute_r_squared_f64(__global const double* y, __global const double* yPred, __global const double* yMean, __global const uint* nSamples, __global double* rSquared) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdataRes[256];
  __local double sdataTot[256];
  uint n = nSamples[0];
  double meanVal = yMean[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    double yVal = y[gid];
    double yPredVal = yPred[gid];
    double resid = (yVal - yPredVal);
    double residSq = (resid * resid);
    double dev = (yVal - meanVal);
    double devSq = (dev * dev);
    sdataRes[lid] = residSq;
    sdataTot[lid] = devSq;
  } else {
    sdataRes[lid] = as_double(0x0000000000000000ul);
    sdataTot[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint r2F64Stride = (uint)(128u);
  for (int reductionStep4 = 0; reductionStep4 < (int)(8); ++reductionStep4) {
    uint stride4 = r2F64Stride;
    if ((lid < stride4)) {
      double otherRes = sdataRes[(lid + stride4)];
      double mineRes = sdataRes[lid];
      double sumRes = (mineRes + otherRes);
      sdataRes[lid] = sumRes;
      double otherTot = sdataTot[(lid + stride4)];
      double mineTot = sdataTot[lid];
      double sumTot = (mineTot + otherTot);
      sdataTot[lid] = sumTot;
    }
    uint strideToHalve4 = r2F64Stride;
    uint nextStride4 = (strideToHalve4 >> (uint)(1u));
    r2F64Stride = nextStride4;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    double ssRes = sdataRes[0];
    double ssTot = sdataTot[0];
    double ratio = (ssRes / ssTot);
    double r2 = (as_double(0x3ff0000000000000ul) - ratio);
    rSquared[0] = r2;
  }
}
