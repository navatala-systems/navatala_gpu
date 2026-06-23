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
extern "C" __global__ void navatala_sparse_sparse_cosine_distance_f32(const float* a, const float* b, const unsigned int* nnz, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ float sdataDot[256];
  __shared__ float sdataNormA[256];
  __shared__ float sdataNormB[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lidU32 = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nnzVal = nnz[0u];
  float localDotAccum = __uint_as_float(0x00000000u);
  float localNormASqAccum = __uint_as_float(0x00000000u);
  float localNormBSqAccum = __uint_as_float(0x00000000u);
  unsigned int idxAccum = gid;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    unsigned int currentIdx = idxAccum;
    if (currentIdx < nnzVal) {
      float aVal = a[currentIdx];
      float bVal = b[currentIdx];
      float prod = (aVal * bVal);
      float currentDot = localDotAccum;
      float newDot = (currentDot + prod);
      localDotAccum = newDot;
      float aSq = (aVal * aVal);
      float bSq = (bVal * bVal);
      float currentNormASq = localNormASqAccum;
      float newNormASq = (currentNormASq + aSq);
      localNormASqAccum = newNormASq;
      float currentNormBSq = localNormBSqAccum;
      float newNormBSq = (currentNormBSq + bSq);
      localNormBSqAccum = newNormBSq;
    }
    unsigned int idxForUpdate = idxAccum;
    unsigned int nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  float finalLocalDot = localDotAccum;
  float finalLocalNormASq = localNormASqAccum;
  float finalLocalNormBSq = localNormBSqAccum;
  sdataDot[lidU32] = finalLocalDot;
  sdataNormA[lidU32] = finalLocalNormASq;
  sdataNormB[lidU32] = finalLocalNormBSq;
  __syncthreads();
  unsigned int strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int currentStride = strideAccum;
    if (lidU32 < currentStride) {
      unsigned int otherIdx = (lidU32 + currentStride);
      float myDot = sdataDot[lidU32];
      float otherDot = sdataDot[otherIdx];
      sdataDot[lidU32] = (myDot + otherDot);
      float myNormA = sdataNormA[lidU32];
      float otherNormA = sdataNormA[otherIdx];
      sdataNormA[lidU32] = (myNormA + otherNormA);
      float myNormB = sdataNormB[lidU32];
      float otherNormB = sdataNormB[otherIdx];
      sdataNormB[lidU32] = (myNormB + otherNormB);
    }
    unsigned int strideForUpdate = strideAccum;
    unsigned int nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    __syncthreads();
  }
  if (lidU32 == 0u) {
    float dotProduct = sdataDot[0u];
    float normASq = sdataNormA[0u];
    float normBSq = sdataNormB[0u];
    float normA = sqrt(normASq);
    float normB = sqrt(normBSq);
    float normProd = (normA * normB);
    bool normIsZero = (normProd == __uint_as_float(0x00000000u));
    float cosineSim = ((normIsZero) ? (__uint_as_float(0x00000000u)) : ((dotProduct / normProd)));
    float cosineDist = (__uint_as_float(0x3f800000u) - cosineSim);
    float finalResult = ((normIsZero) ? (__uint_as_float(0x00000000u)) : (cosineDist));
    result[0u] = finalResult;
  }
}
