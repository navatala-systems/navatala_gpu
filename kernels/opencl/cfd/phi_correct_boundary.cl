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

__kernel void navatala_cfd_phi_correct_boundary(__global const float* phiIn, __global const float* pCell, __global const int* owner, __global const float* magSf, __global const float* deltaCoeffs, __global const float* coeff, __global const float* bcVal, __global const int* bcMask, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* phiOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int f = (counts[1] + ((int)((int)(get_global_id(0)))));
  if ((f >= counts[0])) {
    return;
  } else {
    int o = owner[f];
    float po = pCell[o];
    float other = po;
    if ((bcMask[f] != 0)) {
      other = bcVal[f];
    }
    float snGrad = (deltaCoeffs[f] * (other - po));
    if ((bcSnGradMask[f] != 0)) {
      snGrad = bcSnGrad[f];
    }
    float gradFace = (magSf[f] * snGrad);
    phiOut[f] = (phiIn[f] - ((paramsF[0] * (coeff[f] * gradFace)) + faceFluxCorrection[f]));
  }
}
