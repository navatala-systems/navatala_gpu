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
extern "C" __global__ void navatala_ml_compute_inertia_f64(const double* data, const double* centroids, const unsigned int* assignments, const unsigned int* n, const unsigned int* k, const unsigned int* d, double* inertia) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int dVal = d[0];
  bool inBounds = (gid < nVal);
  double pointDistAccum = __longlong_as_double(0x0000000000000000ull);
  double distSumAccum = __longlong_as_double(0x0000000000000000ull);
  if (inBounds) {
    unsigned int cluster = assignments[gid];
    unsigned int dataBase = (gid * dVal);
    unsigned int centBase = (cluster * dVal);
    distSumAccum = __longlong_as_double(0x0000000000000000ull);
    for (int d = 0; d < (int)(dVal); ++d) {
      unsigned int dU32 = ((unsigned int)(d));
      unsigned int dataIdx = (dataBase + dU32);
      unsigned int centIdx = (centBase + dU32);
      double dataVal = data[dataIdx];
      double centVal = centroids[centIdx];
      double diff = (dataVal - centVal);
      double diffSq = (diff * diff);
      double currDistSum = distSumAccum;
      double newDistSum = (currDistSum + diffSq);
      distSumAccum = newDistSum;
    }
    double distSum = distSumAccum;
    pointDistAccum = distSum;
  }
  double pointDist = pointDistAccum;
  sdata[lid] = pointDist;
  __syncthreads();
  unsigned int inertiaF64RedStride = 128u;
  for (int inertiaF64RedStep = 0; inertiaF64RedStep < (int)(8); ++inertiaF64RedStep) {
    unsigned int inertiaF64Stride = inertiaF64RedStride;
    if (lid < inertiaF64Stride) {
      double other = sdata[(lid + inertiaF64Stride)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int inertiaF64StrideToHalve = inertiaF64RedStride;
    unsigned int inertiaF64NextStride = (inertiaF64StrideToHalve >> 1u);
    inertiaF64RedStride = inertiaF64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    inertia[0] = sdata[0];
  }
}
