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

kernel void navatala_cfd_vof_mules_init(device const float* alpha [[buffer(0)]], device const float* alphaF [[buffer(1)]], device const float* phiBD [[buffer(2)]], device const float* phiCorr [[buffer(3)]], device const int* offsets [[buffer(4)]], device const int* faceIdx [[buffer(5)]], device const float* sign [[buffer(6)]], device const int* owner [[buffer(7)]], device const int* nei [[buffer(8)]], device const float* vol [[buffer(9)]], device const int* counts [[buffer(10)]], device const float* paramsF [[buffer(11)]], device const float* rSubDeltaT [[buffer(12)]], device float* psiMaxCap [[buffer(13)]], device float* psiMinCap [[buffer(14)]], device float* sumPhip [[buffer(15)]], device float* mSumPhim [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float psi = alpha[((int)(int(__gid.x)))];
    if (psi < as_type<float>(0x00000000u)) {
      psi = as_type<float>(0x00000000u);
    }
    if (psi > as_type<float>(0x3f800000u)) {
      psi = as_type<float>(0x3f800000u);
    }
    float maxN = as_type<float>(0x00000000u);
    float minN = as_type<float>(0x3f800000u);
    float sumPhiBD = as_type<float>(0x00000000u);
    float sp = as_type<float>(0x00000000u);
    float sm = as_type<float>(0x00000000u);
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
      if (v < as_type<float>(0x00000000u)) {
        v = as_type<float>(0x00000000u);
      }
      if (v > as_type<float>(0x3f800000u)) {
        v = as_type<float>(0x3f800000u);
      }
      if (v > maxN) {
        maxN = v;
      }
      if (v < minN) {
        minN = v;
      }
      float termBD = (s * phiBD[f]);
      sumPhiBD = (sumPhiBD + termBD);
      float pc = phiCorr[f];
      if (f < counts[2]) {
        if (s > as_type<float>(0x00000000u)) {
          if (pc > as_type<float>(0x00000000u)) {
            sp = (sp + pc);
          } else {
            sm = (sm + (-pc));
          }
        } else {
          if (pc > as_type<float>(0x00000000u)) {
            sm = (sm + pc);
          } else {
            sp = (sp + (-pc));
          }
        }
      } else {
        if (pc > as_type<float>(0x00000000u)) {
          sp = (sp + pc);
        } else {
          sm = (sm + (-pc));
        }
      }
    }
    maxN = (maxN + paramsF[2]);
    if (maxN > as_type<float>(0x3f800000u)) {
      maxN = as_type<float>(0x3f800000u);
    }
    minN = (minN - paramsF[2]);
    if (minN < as_type<float>(0x00000000u)) {
      minN = as_type<float>(0x00000000u);
    }
    if (paramsF[3] > as_type<float>(0x00000000u)) {
      float omSmooth = (as_type<float>(0x3f800000u) - paramsF[3]);
      maxN = ((paramsF[3] * psi) + (omSmooth * maxN));
      if (maxN > as_type<float>(0x3f800000u)) {
        maxN = as_type<float>(0x3f800000u);
      }
      minN = ((paramsF[3] * psi) + (omSmooth * minN));
      if (minN < as_type<float>(0x00000000u)) {
        minN = as_type<float>(0x00000000u);
      }
    }
    sumPhip[((int)(int(__gid.x)))] = sp;
    mSumPhim[((int)(int(__gid.x)))] = sm;
    float V = vol[((int)(int(__gid.x)))];
    psiMaxCap[((int)(int(__gid.x)))] = (((V * rSubDeltaT[((int)(int(__gid.x)))]) * (maxN - psi)) + sumPhiBD);
    psiMinCap[((int)(int(__gid.x)))] = (((V * rSubDeltaT[((int)(int(__gid.x)))]) * (psi - minN)) - sumPhiBD);
  }
}
