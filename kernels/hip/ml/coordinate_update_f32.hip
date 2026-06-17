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
extern "C" __global__ void navatala_ml_coordinate_update_f32(const float* X, const float* residuals, const unsigned int* j, const float* lambda, const unsigned int* nSamples, const unsigned int* nFeatures, float* newBetaJ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdataDot[256];
  __shared__ float sdataNorm[256];
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  unsigned int jVal = j[0];
  float lambdaVal = lambda[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int xijIdx = ((gid * d) + jVal);
    float xij = X[xijIdx];
    float ri = residuals[gid];
    float dotProd = (xij * ri);
    float xijSq = (xij * xij);
    sdataDot[lid] = dotProd;
    sdataNorm[lid] = xijSq;
  } else {
    sdataDot[lid] = __uint_as_float(0x00000000u);
    sdataNorm[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int cu32RedStride = 128u;
  for (int cu32RedStep = 0; cu32RedStep < (int)(8); ++cu32RedStep) {
    unsigned int cu32RedStrideVal = cu32RedStride;
    if ((lid < cu32RedStrideVal)) {
      float otherDot = sdataDot[(lid + cu32RedStrideVal)];
      float mineDot = sdataDot[lid];
      float sumDot = (mineDot + otherDot);
      sdataDot[lid] = sumDot;
      float otherNorm = sdataNorm[(lid + cu32RedStrideVal)];
      float mineNorm = sdataNorm[lid];
      float sumNorm = (mineNorm + otherNorm);
      sdataNorm[lid] = sumNorm;
    }
    unsigned int cu32RedStrideToHalve = cu32RedStride;
    unsigned int cu32RedNextStride = (cu32RedStrideToHalve >> 1u);
    cu32RedStride = cu32RedNextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float totalDot = sdataDot[0];
    float totalNorm = sdataNorm[0];
    float rho = (totalDot / totalNorm);
    float thresh = (lambdaVal / totalNorm);
    float absRho = abs(rho);
    float diff = (absRho - thresh);
    float maxDiff = (((diff > __uint_as_float(0x00000000u))) ? (diff) : (__uint_as_float(0x00000000u)));
    bool positive = (rho > __uint_as_float(0x00000000u));
    bool negative = (rho < __uint_as_float(0x00000000u));
    float signVal = ((positive) ? (__uint_as_float(0x3f800000u)) : (((negative) ? (__uint_as_float(0xbf800000u)) : (__uint_as_float(0x00000000u)))));
    float newBeta = (signVal * maxDiff);
    newBetaJ[0] = newBeta;
  }
}
