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

__kernel void navatala_cfd_stf_all_faces(__global const float* divN, __global const float* alpha, __global const float* alphaF, __global const float* delta, __global const int* owner, __global const int* nei, __global const float* weights, __global const float* contactGrad, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global float* stf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float w = weights[((int)((int)(get_global_id(0))))];
    float wm = (as_float(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (as_float(0xbf800000u) * divInterp);
    float snGrad = as_float(0x00000000u);
    if ((((int)((int)(get_global_id(0)))) < counts[2])) {
      snGrad = (delta[((int)((int)(get_global_id(0))))] * (alpha[n] - alpha[o]));
    } else {
      if ((thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u))) {
        snGrad = contactGrad[((int)((int)(get_global_id(0))))];
      } else {
        snGrad = (delta[((int)((int)(get_global_id(0))))] * (alphaF[((int)((int)(get_global_id(0))))] - alpha[o]));
      }
    }
    stf[((int)((int)(get_global_id(0))))] = ((paramsF[0] * Kf) * snGrad);
  }
}
