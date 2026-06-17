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
extern "C" __global__ void navatala_ml_compute_objective_f64(const double* X, const double* beta, const double* y, const double* lambda, const unsigned int* nSamples, const unsigned int* nFeatures, double* objective) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdataMse[256];
  __shared__ double sdataL1[256];
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  double lambdaVal = lambda[0];
  bool sampleInBounds = (gid < n);
  if (sampleInBounds) {
    double yVal = y[gid];
    unsigned int rowOffset = (gid * d);
    double dotAccum = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int xikIdx = (rowOffset + kU32);
      double xik = X[xikIdx];
      double betak = beta[kU32];
      double prod = (xik * betak);
      double currentDot = dotAccum;
      double newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    double dotProduct = dotAccum;
    double residual = (yVal - dotProduct);
    double residSq = (residual * residual);
    sdataMse[lid] = residSq;
  } else {
    sdataMse[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  bool featureInBounds = (gid < d);
  if (featureInBounds) {
    double betaJ = beta[gid];
    double absBetaJ = abs(betaJ);
    sdataL1[lid] = absBetaJ;
  } else {
    sdataL1[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int obj64RedStride = 128u;
  for (int obj64RedStep = 0; obj64RedStep < (int)(8); ++obj64RedStep) {
    unsigned int obj64RedStrideVal = obj64RedStride;
    if ((lid < obj64RedStrideVal)) {
      double otherMse = sdataMse[(lid + obj64RedStrideVal)];
      double mineMse = sdataMse[lid];
      double sumMse = (mineMse + otherMse);
      sdataMse[lid] = sumMse;
      double otherL1 = sdataL1[(lid + obj64RedStrideVal)];
      double mineL1 = sdataL1[lid];
      double sumL1 = (mineL1 + otherL1);
      sdataL1[lid] = sumL1;
    }
    unsigned int obj64RedStrideToHalve = obj64RedStride;
    unsigned int obj64RedNextStride = (obj64RedStrideToHalve >> 1u);
    obj64RedStride = obj64RedNextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double totalMse = sdataMse[0];
    double totalL1 = sdataL1[0];
    double halfMse = (__longlong_as_double(0x3fe0000000000000ull) * totalMse);
    double lambdaL1 = (lambdaVal * totalL1);
    double obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}
