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
extern "C" __global__ void navatala_sparse_sparse_jaccard_distance_f64(const double* a, const double* b, const unsigned int* nnz, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ unsigned int sdataIntersect[256];
  __shared__ unsigned int sdataUnion[256];
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lidU32 = ((unsigned int)((int)(threadIdx.x)));
  unsigned int nnzVal = nnz[0u];
  unsigned int localIntersectAccum = 0u;
  unsigned int localUnionAccum = 0u;
  unsigned int idxAccum = gid;
  unsigned int gridSize = ((unsigned int)((int)(gridDim.x * blockDim.x)));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    unsigned int currentIdx = idxAccum;
    if ((currentIdx < nnzVal)) {
      double aVal = a[currentIdx];
      double bVal = b[currentIdx];
      bool aNonZero = (aVal != __longlong_as_double(0x0000000000000000ull));
      bool bNonZero = (bVal != __longlong_as_double(0x0000000000000000ull));
      bool inIntersect = (aNonZero && bNonZero);
      bool inUnion = (aNonZero || bNonZero);
      unsigned int intersectInc = ((inIntersect) ? (1u) : (0u));
      unsigned int unionInc = ((inUnion) ? (1u) : (0u));
      unsigned int currentIntersect = localIntersectAccum;
      unsigned int newIntersect = (currentIntersect + intersectInc);
      localIntersectAccum = newIntersect;
      unsigned int currentUnion = localUnionAccum;
      unsigned int newUnion = (currentUnion + unionInc);
      localUnionAccum = newUnion;
    }
    unsigned int idxForUpdate = idxAccum;
    unsigned int nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  unsigned int finalLocalIntersect = localIntersectAccum;
  unsigned int finalLocalUnion = localUnionAccum;
  sdataIntersect[lidU32] = finalLocalIntersect;
  sdataUnion[lidU32] = finalLocalUnion;
  __syncthreads();
  unsigned int strideAccum = 128u;
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    unsigned int currentStride = strideAccum;
    if ((lidU32 < currentStride)) {
      unsigned int otherIdx = (lidU32 + currentStride);
      unsigned int myIntersect = sdataIntersect[lidU32];
      unsigned int otherIntersect = sdataIntersect[otherIdx];
      sdataIntersect[lidU32] = (myIntersect + otherIntersect);
      unsigned int myUnion = sdataUnion[lidU32];
      unsigned int otherUnion = sdataUnion[otherIdx];
      sdataUnion[lidU32] = (myUnion + otherUnion);
    }
    unsigned int strideForUpdate = strideAccum;
    unsigned int nextStride = (strideForUpdate >> 1u);
    strideAccum = nextStride;
    __syncthreads();
  }
  if ((lidU32 == 0u)) {
    unsigned int totalIntersect = sdataIntersect[0u];
    unsigned int totalUnion = sdataUnion[0u];
    bool unionIsZero = (totalUnion == 0u);
    double intersectF = ((double)(totalIntersect));
    double unionF = ((double)(totalUnion));
    double jaccardSim = ((unionIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : ((intersectF / unionF)));
    double jaccardDist = (__longlong_as_double(0x3ff0000000000000ull) - jaccardSim);
    double finalResult = ((unionIsZero) ? (__longlong_as_double(0x0000000000000000ull)) : (jaccardDist));
    result[0u] = finalResult;
  }
}
