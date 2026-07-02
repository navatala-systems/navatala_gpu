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

__kernel void navatala_cfd_vof_cmules_limiter_corr_prepare(__global const float* alpha, __global const float* alphaF, __global const float* phiCorr, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const int* owner, __global const int* nei, __global const float* vol, __global const float* rSubDeltaT, __global const float* sp, __global const float* su, __global const float* psiMax, __global const float* psiMin, __global const int* counts, __global const float* paramsF, __global float* psiMaxCap, __global float* psiMinCap, __global float* sumPhip, __global float* mSumPhim) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)((int)(get_global_id(0))))];
    float psiMaxCell = psiMax[((int)((int)(get_global_id(0))))];
    float psiMinCell = psiMin[((int)((int)(get_global_id(0))))];
    float maxN = psiMinCell;
    float minN = psiMaxCell;
    float spSum = as_float(0x00000000u);
    float smSum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = as_float(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > as_float(0x00000000u)) {
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
        if (s > as_float(0x00000000u)) {
          if (pc > as_float(0x00000000u)) {
            spSum = (spSum + pc);
          } else {
            smSum = (smSum - pc);
          }
        } else {
          if (pc > as_float(0x00000000u)) {
            smSum = (smSum + pc);
          } else {
            spSum = (spSum - pc);
          }
        }
      } else {
        if (pc > as_float(0x00000000u)) {
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
    if (paramsF[3] > as_float(0x00000000u)) {
      float omSmooth = (as_float(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > psiMaxCell) {
        maxN = psiMaxCell;
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < psiMinCell) {
        minN = psiMinCell;
      }
    }
    sumPhip[((int)((int)(get_global_id(0))))] = spSum;
    mSumPhim[((int)((int)(get_global_id(0))))] = smSum;
    float V = vol[((int)((int)(get_global_id(0))))];
    float rdt = rSubDeltaT[((int)((int)(get_global_id(0))))];
    float spv = sp[((int)((int)(get_global_id(0))))];
    float suv = su[((int)((int)(get_global_id(0))))];
    float diag = (rdt - spv);
    psiMaxCap[((int)((int)(get_global_id(0))))] = (V * (((diag * maxN) - suv) - (rdt * psi)));
    psiMinCap[((int)((int)(get_global_id(0))))] = (V * ((suv - (diag * minN)) + (rdt * psi)));
  }
}
