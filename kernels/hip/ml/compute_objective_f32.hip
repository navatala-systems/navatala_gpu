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
extern "C" __global__ void navatala_ml_compute_objective_f32(const float* X, const float* beta, const float* y, const float* lambda, const unsigned int* nSamples, const unsigned int* nFeatures, float* objective) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdataMse[256];
  __shared__ float sdataL1[256];
  unsigned int n = nSamples[0];
  unsigned int d = nFeatures[0];
  float lambdaVal = lambda[0];
  bool sampleInBounds = (gid < n);
  if (sampleInBounds) {
    float yVal = y[gid];
    unsigned int rowOffset = (gid * d);
    float dotAccum = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int xikIdx = (rowOffset + kU32);
      float xik = X[xikIdx];
      float betak = beta[kU32];
      float prod = (xik * betak);
      float currentDot = dotAccum;
      float newDot = (currentDot + prod);
      dotAccum = newDot;
    }
    float dotProduct = dotAccum;
    float residual = (yVal - dotProduct);
    float residSq = (residual * residual);
    sdataMse[lid] = residSq;
  } else {
    sdataMse[lid] = __uint_as_float(0x00000000u);
  }
  bool featureInBounds = (gid < d);
  if (featureInBounds) {
    float betaJ = beta[gid];
    float absBetaJ = abs(betaJ);
    sdataL1[lid] = absBetaJ;
  } else {
    sdataL1[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int obj32RedStride = 128u;
  for (int obj32RedStep = 0; obj32RedStep < (int)(8); ++obj32RedStep) {
    unsigned int obj32RedStrideVal = obj32RedStride;
    if (lid < obj32RedStrideVal) {
      float otherMse = sdataMse[(lid + obj32RedStrideVal)];
      float mineMse = sdataMse[lid];
      float sumMse = (mineMse + otherMse);
      sdataMse[lid] = sumMse;
      float otherL1 = sdataL1[(lid + obj32RedStrideVal)];
      float mineL1 = sdataL1[lid];
      float sumL1 = (mineL1 + otherL1);
      sdataL1[lid] = sumL1;
    }
    unsigned int obj32RedStrideToHalve = obj32RedStride;
    unsigned int obj32RedNextStride = (obj32RedStrideToHalve >> 1u);
    obj32RedStride = obj32RedNextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float totalMse = sdataMse[0];
    float totalL1 = sdataL1[0];
    float halfMse = (__uint_as_float(0x3f000000u) * totalMse);
    float lambdaL1 = (lambdaVal * totalL1);
    float obj = (halfMse + lambdaL1);
    objective[0] = obj;
  }
}
