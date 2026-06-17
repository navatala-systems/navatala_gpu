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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_sparse_cosine_distance_f64(__global const double* a, __global const double* b, __global const uint* nnz, __global double* result) {
  int gid0 = (int)get_global_id(0);
  __local double sdataDot[256];
  __local double sdataNormA[256];
  __local double sdataNormB[256];
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lidU32 = ((uint)((int)(get_local_id(0))));
  uint nnzVal = nnz[(uint)(0u)];
  double localDotAccum = as_double(0x0000000000000000ul);
  double localNormASqAccum = as_double(0x0000000000000000ul);
  double localNormBSqAccum = as_double(0x0000000000000000ul);
  uint idxAccum = gid;
  uint gridSize = ((uint)((int)(get_global_size(0))));
  for (int loopIdx = 0; loopIdx < (int)(nnzVal); ++loopIdx) {
    uint currentIdx = idxAccum;
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
    uint idxForUpdate = idxAccum;
    uint nextIdx = (idxForUpdate + gridSize);
    idxAccum = nextIdx;
  }
  double finalLocalDot = localDotAccum;
  double finalLocalNormASq = localNormASqAccum;
  double finalLocalNormBSq = localNormBSqAccum;
  sdataDot[lidU32] = finalLocalDot;
  sdataNormA[lidU32] = finalLocalNormASq;
  sdataNormB[lidU32] = finalLocalNormBSq;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint strideAccum = (uint)(128u);
  for (int reductionStep = 0; reductionStep < (int)(8); ++reductionStep) {
    uint currentStride = strideAccum;
    if ((lidU32 < currentStride)) {
      uint otherIdx = (lidU32 + currentStride);
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
    uint strideForUpdate = strideAccum;
    uint nextStride = (strideForUpdate >> (uint)(1u));
    strideAccum = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lidU32 == (uint)(0u))) {
    double dotProduct = sdataDot[(uint)(0u)];
    double normASq = sdataNormA[(uint)(0u)];
    double normBSq = sdataNormB[(uint)(0u)];
    double normA = sqrt(normASq);
    double normB = sqrt(normBSq);
    double normProd = (normA * normB);
    bool normIsZero = (normProd == as_double(0x0000000000000000ul));
    double cosineSim = ((normIsZero) ? (as_double(0x0000000000000000ul)) : ((dotProduct / normProd)));
    double cosineDist = (as_double(0x3ff0000000000000ul) - cosineSim);
    double finalResult = ((normIsZero) ? (as_double(0x0000000000000000ul)) : (cosineDist));
    result[(uint)(0u)] = finalResult;
  }
}
