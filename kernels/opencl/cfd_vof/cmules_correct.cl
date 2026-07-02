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

__kernel void navatala_cfd_vof_cmules_correct(__global const float* alphaOld, __global const float* rSubDeltaT, __global const float* sp, __global const float* su, __global const float* phiCorrLimited, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* counts, __global const float* paramsF, __global float* alphaNext) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float acc = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float term = (sign[k] * phiCorrLimited[f]);
      acc = (acc + term);
    }
    float divCorr = (acc / vol[((int)((int)(get_global_id(0))))]);
    float rdt = rSubDeltaT[((int)((int)(get_global_id(0))))];
    float spv = sp[((int)((int)(get_global_id(0))))];
    float suv = su[((int)((int)(get_global_id(0))))];
    float den = (rdt - spv);
    alphaNext[((int)((int)(get_global_id(0))))] = ((((alphaOld[((int)((int)(get_global_id(0))))] * rdt) + suv) - divCorr) / den);
  }
}
