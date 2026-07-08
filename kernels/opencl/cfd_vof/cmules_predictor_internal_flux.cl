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

__kernel void navatala_cfd_vof_cmules_predictor_internal_flux(__global const float* alphaPred, __global const float* phiCN, __global const int* owner, __global const int* nei, __global const int* counts, __global float* outPhiUD) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[2])) > 0 ? ((int)(counts[2])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[2]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[2]) {
    return;
  } else {
    float ph = phiCN[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float a = as_float(0x00000000u);
    if (ph >= as_float(0x00000000u)) {
      a = alphaPred[o];
    } else {
      a = alphaPred[n];
    }
    outPhiUD[((int)((int)(get_global_id(0))))] = (ph * a);
  }
}
