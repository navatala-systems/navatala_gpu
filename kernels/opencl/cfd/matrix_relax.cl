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

__kernel void navatala_cfd_matrix_relax(__global const float* diagOld, __global const float* sumOff, __global const float* psi, __global const int* counts, __global const float* paramsF, __global float* diag, __global float* source) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float d0 = diagOld[((int)((int)(get_global_id(0))))];
    float sOff = sumOff[((int)((int)(get_global_id(0))))];
    float x = psi[((int)((int)(get_global_id(0))))];
    float negD0 = (as_float(0x00000000u) - d0);
    float absD0 = (((d0 >= as_float(0x00000000u))) ? (d0) : (negD0));
    float maxVal = (((absD0 >= sOff)) ? (absD0) : (sOff));
    float newDiag = (maxVal / paramsF[0]);
    float srcPrev = source[((int)((int)(get_global_id(0))))];
    float delta = (newDiag - d0);
    float srcNew = (srcPrev + (delta * x));
    diag[((int)((int)(get_global_id(0))))] = newDiag;
    source[((int)((int)(get_global_id(0))))] = srcNew;
  }
}
