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
__kernel void navatala_ml_coordinate_update_f64(__global const double* X, __global const double* residuals, __global const uint* j, __global const double* lambda, __global const uint* nSamples, __global const uint* nFeatures, __global double* newBetaJ) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdataDot[256];
  __local double sdataNorm[256];
  uint n = nSamples[0];
  uint d = nFeatures[0];
  uint jVal = j[0];
  double lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint xijIdx = ((gid * d) + jVal);
    double xij = X[xijIdx];
    double ri = residuals[gid];
    double dotProd = (xij * ri);
    double xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = as_double(0x0000000000000000ul);
    sdataNorm[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint cu64RedStride = (uint)(128u);
  for (int cu64RedStep = 0; cu64RedStep < (int)(8); ++cu64RedStep) {
    uint cu64RedStrideVal = cu64RedStride;
    if (lid < cu64RedStrideVal) {
      double otherDot = sdataDot[(lid + cu64RedStrideVal)];
      double mineDot = sdataDot[lid];
      double sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      double otherNorm = sdataNorm[(lid + cu64RedStrideVal)];
      double mineNorm = sdataNorm[lid];
      double sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    uint cu64RedStrideToHalve = cu64RedStride;
    uint cu64RedNextStride = (cu64RedStrideToHalve >> (uint)(1u));
    cu64RedStride = cu64RedNextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double totalDot = sdataDot[0];
    double totalNorm = sdataNorm[0];
    double rho = (totalDot / totalNorm);
    double thresh = (lambdaVal / totalNorm);
    double absRho = fabs(rho);
    double diff = (absRho - thresh);
    double maxDiff = (((diff > as_double(0x0000000000000000ul))) ? (diff) : (as_double(0x0000000000000000ul)));
    bool positive = (rho > as_double(0x0000000000000000ul));
    bool negative = (rho < as_double(0x0000000000000000ul));
    double signVal = ((positive) ? (as_double(0x3ff0000000000000ul)) : (((negative) ? (as_double(0xbff0000000000000ul)) : (as_double(0x0000000000000000ul)))));
    double newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}
