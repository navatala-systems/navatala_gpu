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

__kernel void navatala_cfd_laplacian_scalar_explicit(__global const float* muCell, __global const float* phiCell, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* magSf, __global const float* deltaCoeffs, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const float* vol, __global const float* bcMu, __global const uint* bcMuMask, __global const float* bcPhi, __global const uint* bcPhiMask, __global const int* params, __global float* outL) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float phiC = phiCell[((int)((int)(get_global_id(0))))];
    float muC = muCell[((int)((int)(get_global_id(0))))];
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        int sgn = sign[k];
        int o = owner[f];
        float phiOther = phiCell[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            phiOther = phiCell[n];
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if (bcPhiMask[f] != (uint)(0u)) {
              phiOther = bcPhi[f];
            }
            if (bcMuMask[f] != (uint)(0u)) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if (v != as_float(0x00000000u)) {
      invV = (as_float(0x3f800000u) / v);
    }
    outL[((int)((int)(get_global_id(0))))] = (sum * invV);
  }
}
