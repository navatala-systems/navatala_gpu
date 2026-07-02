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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_vof_cmules_limiter_corr_prepare(device const float* alpha [[buffer(0)]], device const float* alphaF [[buffer(1)]], device const float* phiCorr [[buffer(2)]], device const int* offsets [[buffer(3)]], device const int* faceIdx [[buffer(4)]], device const float* sign [[buffer(5)]], device const int* owner [[buffer(6)]], device const int* nei [[buffer(7)]], device const float* vol [[buffer(8)]], device const float* rSubDeltaT [[buffer(9)]], device const float* sp [[buffer(10)]], device const float* su [[buffer(11)]], device const float* psiMax [[buffer(12)]], device const float* psiMin [[buffer(13)]], device const int* counts [[buffer(14)]], device const float* paramsF [[buffer(15)]], device float* psiMaxCap [[buffer(16)]], device float* psiMinCap [[buffer(17)]], device float* sumPhip [[buffer(18)]], device float* mSumPhim [[buffer(19)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)(int(__gid.x)))];
    float psiMaxCell = psiMax[((int)(int(__gid.x)))];
    float psiMinCell = psiMin[((int)(int(__gid.x)))];
    float maxN = psiMinCell;
    float minN = psiMaxCell;
    float spSum = as_type<float>(0x00000000u);
    float smSum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float v = as_type<float>(0x00000000u);
      if (f < counts[2]) {
        int nbr = owner[f];
        if (s > as_type<float>(0x00000000u)) {
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
        if (s > as_type<float>(0x00000000u)) {
          if (pc > as_type<float>(0x00000000u)) {
            spSum = (spSum + pc);
          } else {
            smSum = (smSum - pc);
          }
        } else {
          if (pc > as_type<float>(0x00000000u)) {
            smSum = (smSum + pc);
          } else {
            spSum = (spSum - pc);
          }
        }
      } else {
        if (pc > as_type<float>(0x00000000u)) {
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
    if (paramsF[3] > as_type<float>(0x00000000u)) {
      float omSmooth = (as_type<float>(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > psiMaxCell) {
        maxN = psiMaxCell;
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < psiMinCell) {
        minN = psiMinCell;
      }
    }
    sumPhip[((int)(int(__gid.x)))] = spSum;
    mSumPhim[((int)(int(__gid.x)))] = smSum;
    float V = vol[((int)(int(__gid.x)))];
    float rdt = rSubDeltaT[((int)(int(__gid.x)))];
    float spv = sp[((int)(int(__gid.x)))];
    float suv = su[((int)(int(__gid.x)))];
    float diag = (rdt - spv);
    psiMaxCap[((int)(int(__gid.x)))] = (V * (((diag * maxN) - suv) - (rdt * psi)));
    psiMinCap[((int)(int(__gid.x)))] = (V * ((suv - (diag * minN)) + (rdt * psi)));
  }
}
