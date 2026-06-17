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
extern "C" __global__ void navatala_cfd_vof_mules_init(const float* alpha, const float* alphaF, const float* phiBD, const float* phiCorr, const int* offsets, const int* faceIdx, const float* sign, const int* owner, const int* nei, const float* vol, const int* counts, const float* paramsF, const float* rSubDeltaT, float* psiMaxCap, float* psiMinCap, float* sumPhip, float* mSumPhim) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0])) {
    return;
  } else {
    float psi = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if ((psi < __uint_as_float(0x00000000u))) {
      psi = __uint_as_float(0x00000000u);
    }
    if ((psi > __uint_as_float(0x3f800000u))) {
      psi = __uint_as_float(0x3f800000u);
    }
    float maxN = __uint_as_float(0x00000000u);
    float minN = __uint_as_float(0x3f800000u);
    float sumPhiBD = __uint_as_float(0x00000000u);
    float sp = __uint_as_float(0x00000000u);
    float sm = __uint_as_float(0x00000000u);
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = __uint_as_float(0x00000000u);
      if ((f < counts[2])) {
        int nbr = owner[f];
        if ((s > __uint_as_float(0x00000000u))) {
          nbr = nei[f];
        }
        v = alpha[nbr];
      } else {
        v = alphaF[f];
      }
      if ((v < __uint_as_float(0x00000000u))) {
        v = __uint_as_float(0x00000000u);
      }
      if ((v > __uint_as_float(0x3f800000u))) {
        v = __uint_as_float(0x3f800000u);
      }
      if ((v > maxN)) {
        maxN = v;
      }
      if ((v < minN)) {
        minN = v;
      }
      float termBD = (s * phiBD[f]);
      sumPhiBD = (sumPhiBD + termBD);
      float pc = phiCorr[f];
      if ((f < counts[2])) {
        if ((s > __uint_as_float(0x00000000u))) {
          if ((pc > __uint_as_float(0x00000000u))) {
            sp = (sp + pc);
          } else {
            sm = (sm + (-pc));
          }
        } else {
          if ((pc > __uint_as_float(0x00000000u))) {
            sm = (sm + pc);
          } else {
            sp = (sp + (-pc));
          }
        }
      } else {
        if ((pc > __uint_as_float(0x00000000u))) {
          sp = (sp + pc);
        } else {
          sm = (sm + (-pc));
        }
      }
    }
    maxN = (maxN + paramsF[2]);
    if ((maxN > __uint_as_float(0x3f800000u))) {
      maxN = __uint_as_float(0x3f800000u);
    }
    minN = (minN - paramsF[2]);
    if ((minN < __uint_as_float(0x00000000u))) {
      minN = __uint_as_float(0x00000000u);
    }
    if ((paramsF[3] > __uint_as_float(0x00000000u))) {
      float omSmooth = (__uint_as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if ((maxN > __uint_as_float(0x3f800000u))) {
        maxN = __uint_as_float(0x3f800000u);
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if ((minN < __uint_as_float(0x00000000u))) {
        minN = __uint_as_float(0x00000000u);
      }
    }
    sumPhip[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sp;
    mSumPhim[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = sm;
    float V = vol[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    psiMaxCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((V * rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) * (maxN - psi)) + sumPhiBD);
    psiMinCap[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (((V * rSubDeltaT[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]) * (psi - minN)) - sumPhiBD);
  }
}
