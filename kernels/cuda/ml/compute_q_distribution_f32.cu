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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_ml_compute_q_distribution_f32(const float* Y, const unsigned int* nPoints, const unsigned int* outputDims, float* Q, float* sumQ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = outputDims[0];
  unsigned int totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  float localQ = __uint_as_float(0x00000000u);
  if (inBounds) {
    unsigned int i = (gid / n);
    unsigned int j = (gid % n);
    unsigned int baseI = (i * d);
    unsigned int baseJ = (j * d);
    float distSqAccum = __uint_as_float(0x00000000u);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int idxI = (baseI + kU32);
      unsigned int idxJ = (baseJ + kU32);
      float yi_k = Y[idxI];
      float yj_k = Y[idxJ];
      float diff = (yi_k - yj_k);
      float diffSq = (diff * diff);
      float currentDistSq = distSqAccum;
      float newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    float distSq = distSqAccum;
    bool isDiag = (i == j);
    float onePlusDistSq = (__uint_as_float(0x3f800000u) + distSq);
    float qVal = (__uint_as_float(0x3f800000u) / onePlusDistSq);
    if (isDiag) {
      Q[gid] = __uint_as_float(0x00000000u);
      sdata[lid] = __uint_as_float(0x00000000u);
    } else {
      Q[gid] = qVal;
      sdata[lid] = qVal;
    }
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int qF32ReductionStride = 128u;
  for (int qF32ReductionStep = 0; qF32ReductionStep < (int)(8); ++qF32ReductionStep) {
    unsigned int qF32Stride = qF32ReductionStride;
    if (lid < qF32Stride) {
      float qF32Other = sdata[(lid + qF32Stride)];
      float qF32Mine = sdata[lid];
      float qF32Sum = (qF32Mine + qF32Other);
      sdata[lid] = qF32Sum;
    }
    unsigned int qF32StrideToHalve = qF32ReductionStride;
    unsigned int qF32NextStride = (qF32StrideToHalve >> 1u);
    qF32ReductionStride = qF32NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float partialSum = sdata[0];
    float oldSum = sumQ[0];
    float newSum = (oldSum + partialSum);
    sumQ[0] = newSum;
  }
}
