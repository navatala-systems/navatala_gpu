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

__kernel void navatala_cfd_phi_correct_internal(__global const float* phiIn, __global const float* pCell, __global const int* owner, __global const int* neighbour, __global const float* magSf, __global const float* deltaCoeffs, __global const float* coeff, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* phiOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[1])) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = neighbour[((int)((int)(get_global_id(0))))];
    float po = pCell[o];
    float pn = pCell[n];
    float snGrad = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (pn - po));
    float gradFace = (magSf[((int)((int)(get_global_id(0))))] * snGrad);
    phiOut[((int)((int)(get_global_id(0))))] = (phiIn[((int)((int)(get_global_id(0))))] - ((paramsF[0] * (coeff[((int)((int)(get_global_id(0))))] * gradFace)) + faceFluxCorrection[((int)((int)(get_global_id(0))))]));
  }
}
