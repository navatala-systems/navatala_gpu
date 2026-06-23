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

__kernel void navatala_cfd_linear_upwind_correction_source(__global const float* luFaceVals, __global const float* flux, __global const int* owner, __global const int* neighbour, __global const float* psi, __global const int* offsets, __global const int* faceIdx, __global const float* signF, __global const int* counts, __global float* outSource) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < counts[1])) {
        float fluxVal = flux[f];
        float luVal = luFaceVals[f];
        float sgnF = signF[k];
        float correction = (fluxVal * luVal);
        sum = (sum + (sgnF * correction));
      }
    }
    float prev = outSource[((int)((int)(get_global_id(0))))];
    outSource[((int)((int)(get_global_id(0))))] = (prev - sum);
  }
}
