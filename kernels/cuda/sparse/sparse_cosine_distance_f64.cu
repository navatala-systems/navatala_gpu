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
extern "C" __global__ void navatala_sparse_sparse_cosine_distance_f64(const double* a, const double* b, const unsigned int* nnz, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double sdataDot[256];
  __shared__ double sdataNormA[256];
  __shared__ double sdataNormB[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lidU32 = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nnzVal = nnz[0u];
  double localDotAccum = __longlong_as_double(0x0000000000000000ull);
  double localNormASqAccum = __longlong_as_double(0x0000000000000000ull);
  double localNormBSqAccum = __longlong_as_double(0x0000000000000000ull);
  unsigned int idxAccum = gid;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    unsigned int currentIdx = idxAccum;
    if ((currentIdx < nnzVal)) {
      double aVal = a[currentIdx];
      double bVal = b[currentIdx];
      double prod = (aVal * bVal);
      double currentDot = localDotAccum;
      double newDot = (currentDot + prod);
      localDotAccum = newDot;
      double aSq = (aVal * aVal);
      double bSq = (bVal * bVal);
      double currentNormASq = localNormASqAccum;
      double newNormASq = (currentNormASq + aSq);
      localNormASqAccum = newNormASq;
      double currentNormBSq = localNormBSqAccum;
      double newNormBSq = (currentNormBSq + bSq);
      localNormBSqAccum = newNormBSq;
    }
    unsigned int idxForUpdate = idxAccum;
    unsigned int nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  double finalLocalDot = localDotAccum;
  double finalLocalNormASq = localNormASqAccum;
  double finalLocalNormBSq = localNormBSqAccum;
  sdataDot[lidU32] = finalLocalDot;
  sdataNormA[lidU32] = finalLocalNormASq;
  sdataNormB[lidU32] = finalLocalNormBSq;
  __syncthreads();
  unsigned int strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int currentStride = strideAccum;
    if ((lidU32 < currentStride)) {
      unsigned int otherIdx = (lidU32 + currentStride);
      double myDot = sdataDot[lidU32];
      double otherDot = sdataDot[otherIdx];
      sdataDot[lidU32] = (myDot + otherDot);
      double myNormA = sdataNormA[lidU32];
      double otherNormA = sdataNormA[otherIdx];
      sdataNormA[lidU32] = (myNormA + otherNormA);
      double myNormB = sdataNormB[lidU32];
      double otherNormB = sdataNormB[otherIdx];
      sdataNormB[lidU32] = (myNormB + otherNormB);
    }
    unsigned int strideForUpdate = strideAccum;
    unsigned int nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    __syncthreads();
  }
  if ((lidU32 == 0u)) {
    double dotProduct = sdataDot[0u];
    double normASq = sdataNormA[0u];
    double normBSq = sdataNormB[0u];
    double normA = sqrt(normASq);
    double normB = sqrt(normBSq);
    double normProd = (normA * normB);
    bool normIsZero = (normProd == __longlong_as_double(0x0000000000000000ull));
    double cosineSim = ((normIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : ((dotProduct / normProd)));
    double cosineDist = (__longlong_as_double(0x3ff0000000000000ull) - cosineSim);
    double finalResult = ((normIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : (cosineDist));
    result[0u] = finalResult;
  }
}
