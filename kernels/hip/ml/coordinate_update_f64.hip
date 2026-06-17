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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_coordinate_update_f64(const double* X, const double* residuals, const unsigned int* j, const double* lambda, const unsigned int* nSamples, const unsigned int* nFeatures, double* newBetaJ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdataDot[256];
  __shared__ double sdataNorm[256];
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int jVal = j[0];
  double lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int xijIdx = ((gid * d) + jVal);
    double xij = X[xijIdx];
    double ri = residuals[gid];
    double dotProd = (xij * ri);
    double xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = __longlong_as_double(0x0000000000000000ull);
    sdataNorm[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int cu64RedStride = 128u;
  for (int cu64RedStep = 0; cu64RedStep < (int)(8); ++cu64RedStep) {
    unsigned int cu64RedStrideVal = cu64RedStride;
    if ((lid < cu64RedStrideVal)) {
      double otherDot = sdataDot[(lid + cu64RedStrideVal)];
      double mineDot = sdataDot[lid];
      double sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      double otherNorm = sdataNorm[(lid + cu64RedStrideVal)];
      double mineNorm = sdataNorm[lid];
      double sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    unsigned int cu64RedStrideToHalve = cu64RedStride;
    unsigned int cu64RedNextStride = (cu64RedStrideToHalve >> 1u);
    cu64RedStride = cu64RedNextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double totalDot = sdataDot[0];
    double totalNorm = sdataNorm[0];
    double rho = (totalDot / totalNorm);
    double thresh = (lambdaVal / totalNorm);
    double absRho = abs(rho);
    double diff = (absRho - thresh);
    double maxDiff = (((diff > __longlong_as_double(0x0000000000000000ull))) ? (diff) : (__longlong_as_double(0x0000000000000000ull)));
    bool positive = (rho > __longlong_as_double(0x0000000000000000ull));
    bool negative = (rho < __longlong_as_double(0x0000000000000000ull));
    double signVal = ((positive) ? (__longlong_as_double(0x3ff0000000000000ull)) : (((negative) ? (__longlong_as_double(0xbff0000000000000ull)) : (__longlong_as_double(0x0000000000000000ull)))));
    double newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}
