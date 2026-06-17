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
extern "C" __global__ void navatala_ml_compute_q_distribution_f64(const double* Y, const unsigned int* nPoints, const unsigned int* outputDims, double* Q, double* sumQ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double sdata[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = outputDims[0];
  unsigned int totalPairs = (n * n);
  bool inBounds = (gid < totalPairs);
  double localQ = __longlong_as_double(0x0000000000000000ull);
  if (inBounds) {
    unsigned int i = (gid / n);
    unsigned int j = (gid % n);
    unsigned int baseI = (i * d);
    unsigned int baseJ = (j * d);
    double distSqAccum = __longlong_as_double(0x0000000000000000ull);
    for (int k = 0; k < (int)(d); ++k) {
      unsigned int kU32 = ((unsigned int)(k));
      unsigned int idxI = (baseI + kU32);
      unsigned int idxJ = (baseJ + kU32);
      double yi_k = Y[idxI];
      double yj_k = Y[idxJ];
      double diff = (yi_k - yj_k);
      double diffSq = (diff * diff);
      double currentDistSq = distSqAccum;
      double newDistSq = (currentDistSq + diffSq);
      distSqAccum = newDistSq;
    }
    double distSq = distSqAccum;
    bool isDiag = (i == j);
    double onePlusDistSq = (__longlong_as_double(0x3ff0000000000000ull) + distSq);
    double qVal = (__longlong_as_double(0x3ff0000000000000ull) / onePlusDistSq);
    if (isDiag) {
      Q[gid] = __longlong_as_double(0x0000000000000000ull);
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    } else {
      Q[gid] = qVal;
      sdata[lid] = qVal;
    }
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int qF64ReductionStride = 128u;
  for (int qF64ReductionStep = 0; qF64ReductionStep < (int)(8); ++qF64ReductionStep) {
    unsigned int qF64Stride = qF64ReductionStride;
    if ((lid < qF64Stride)) {
      double qF64Other = sdata[(lid + qF64Stride)];
      double qF64Mine = sdata[lid];
      double qF64Sum = (qF64Mine + qF64Other);
      sdata[lid] = qF64Sum;
    }
    unsigned int qF64StrideToHalve = qF64ReductionStride;
    unsigned int qF64NextStride = (qF64StrideToHalve >> 1u);
    qF64ReductionStride = qF64NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    double partialSum = sdata[0];
    double oldSum = sumQ[0];
    double newSum = (oldSum + partialSum);
    sumQ[0] = newSum;
  }
}
