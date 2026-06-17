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
extern "C" __global__ void navatala_ml_compute_inertia_f32(const float* data, const float* centroids, const unsigned int* assignments, const unsigned int* n, const unsigned int* k, const unsigned int* d, float* inertia) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  bool inBounds = (gid < nVal);
  float pointDistAccum = __uint_as_float(0x00000000u);
  float distSumAccum = __uint_as_float(0x00000000u);
  if (inBounds) {
    unsigned int cluster = assignments[gid];
    unsigned int dataBase = (gid * dVal);
    unsigned int centBase = (cluster * dVal);
    distSumAccum = __uint_as_float(0x00000000u);
    for (int d = 0; d < (int)(dVal); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int dataIdx = (dataBase + dU32);
      unsigned int centIdx = (centBase + dU32);
      float dataVal = data[dataIdx];
      float centVal = centroids[centIdx];
      float diff = (dataVal - centVal);
      float diffSq = (diff * diff);
      float currDistSum = distSumAccum;
      float newDistSum = (currDistSum + diffSq);
      distSumAccum = newDistSum;
    }
    float distSum = distSumAccum;
    pointDistAccum = distSum;
  }
  float pointDist = pointDistAccum;
  sdata[lid] = pointDist;
  __syncthreads();
  unsigned int inertiaF32RedStride = 128u;
  for (int inertiaF32RedStep = 0; inertiaF32RedStep < (int)(8); ++inertiaF32RedStep) {
    unsigned int inertiaF32Stride = inertiaF32RedStride;
    if ((lid < inertiaF32Stride)) {
      float other = sdata[(lid + inertiaF32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int inertiaF32StrideToHalve = inertiaF32RedStride;
    unsigned int inertiaF32NextStride = (inertiaF32StrideToHalve >> 1u);
    inertiaF32RedStride = inertiaF32NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    inertia[0] = sdata[0];
  }
}
