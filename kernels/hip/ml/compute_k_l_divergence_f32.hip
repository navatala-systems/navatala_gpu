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
extern "C" __global__ void navatala_ml_compute_k_l_divergence_f32(const float* P, const float* Q, const unsigned int* nPoints, float* kl_divergence) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  if (inBounds) {
    float pij = P[gid];
    float qij = Q[gid];
    float minVal = __uint_as_float(0x33d6bf95u);
    bool pSafe = (pij > minVal);
    bool qSafe = (qij > minVal);
    bool bothSafe = (pSafe && qSafe);
    if (bothSafe) {
      float ratio = (pij / qij);
      float logRatio = log(ratio);
      float klTerm = (pij * logRatio);
      sdata[lid] = klTerm;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int klF32ReductionStride = 128u;
  for (int klF32ReductionStep = 0; klF32ReductionStep < (int)(8); ++klF32ReductionStep) {
    unsigned int klF32Stride = klF32ReductionStride;
    if ((lid < klF32Stride)) {
      float klF32Other = sdata[(lid + klF32Stride)];
      float klF32Mine = sdata[lid];
      float klF32Sum = (klF32Mine + klF32Other);
      sdata[lid] = klF32Sum;
    }
    unsigned int klF32StrideToHalve = klF32ReductionStride;
    unsigned int klF32NextStride = (klF32StrideToHalve >> 1u);
    klF32ReductionStride = klF32NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    float klVal = sdata[0];
    kl_divergence[0] = klVal;
  }
}
