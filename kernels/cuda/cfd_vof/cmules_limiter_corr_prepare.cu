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
extern "C" __global__ void navatala_cfd_vof_cmules_limiter_corr_prepare(const float* alpha, const float* alphaF, const float* phiCorr, const int* offsets, const int* faceIdx, const float* sign, const int* owner, const int* nei, const float* vol, const float* rSubDeltaT, const float* sp, const float* su, const float* psiMax, const float* psiMin, const int* counts, const float* paramsF, float* psiMaxCap, float* psiMinCap, float* sumPhip, float* mSumPhim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiMaxCell = psiMax[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float psiMinCell = psiMin[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float maxN = psiMinCell;
    float minN = psiMaxCell;
    float spSum = __uint_as_float(0x00000000u);
    float smSum = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = __uint_as_float(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > __uint_as_float(0x00000000u)) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if (v > maxN) {
        maxN = v;
      }
      if (v < minN) {
        minN = v;
      }
      float pc = phiCorr[f];
      if (f < counts[2]) {
        if (s > __uint_as_float(0x00000000u)) {
          if (pc > __uint_as_float(0x00000000u)) {
            spSum = (spSum + pc);
          } else {
            smSum = (smSum - pc);
          }
        } else {
          if (pc > __uint_as_float(0x00000000u)) {
            smSum = (smSum + pc);
          } else {
            spSum = (spSum - pc);
          }
        }
      } else {
        if (pc > __uint_as_float(0x00000000u)) {
          spSum = (spSum + pc);
        } else {
          smSum = (smSum - pc);
        }
      }
    }
    float range = (psiMaxCell - psiMinCell);
    maxN = (maxN + (paramsF[2] * range));
    if (maxN > psiMaxCell) {
      maxN = psiMaxCell;
    }
    minN = (minN - (paramsF[2] * range));
    if (minN < psiMinCell) {
      minN = psiMinCell;
    }
    if (paramsF[3] > __uint_as_float(0x00000000u)) {
      float omSmooth = (__uint_as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > psiMaxCell) {
        maxN = psiMaxCell;
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < psiMinCell) {
        minN = psiMinCell;
      }
    }
    sumPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = spSum;
    mSumPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = smSum;
    float V = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rdt = rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float spv = sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float suv = su[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float diag = (rdt - spv);
    psiMaxCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (V * (((diag * maxN) - suv) - (rdt * psi)));
    psiMinCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (V * ((suv - (diag * minN)) + (rdt * psi)));
  }
}
