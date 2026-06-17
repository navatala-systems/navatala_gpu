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

__kernel void navatala_cfd_vof_mules_face_update(__global const float* phiCorr, __global float* lambda, __global const int* owner, __global const int* nei, __global const float* lambdam, __global const float* lambdap, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[2])) {
    return;
  } else {
    float pc = phiCorr[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float a0 = lambdam[o];
    float b0 = lambdap[n];
    float lim = a0;
    if ((lim > b0)) {
      lim = b0;
    }
    if ((pc > as_float(0x00000000u))) {
      float a1 = lambdap[o];
      float b1 = lambdam[n];
      float lim1 = a1;
      if ((lim1 > b1)) {
        lim1 = b1;
      }
      lim = lim1;
    }
    float cur = lambda[((int)((int)(get_global_id(0))))];
    float _out = cur;
    if ((_out > lim)) {
      _out = lim;
    }
    lambda[((int)((int)(get_global_id(0))))] = _out;
  }
}
