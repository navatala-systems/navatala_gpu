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

__kernel void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(__global const float* rhoCell, __global const float* muCell, __global const float* phiFaces, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const float* muBcVal, __global const uint* muBcMask, __global const int* ownerAll, __global const int* neighbourInt, __global const float* weightsAll, __global const float* magSfAll, __global const float* deltaAll, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const float* volCells, __global const float* sp, __global const int* counts3, __global const float* paramsF, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  if ((((int)((int)(get_global_id(0)))) >= counts3[0])) {
    return;
  } else {
    float invDt = (as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(get_global_id(0))))];
    float A = (rhoCell[((int)((int)(get_global_id(0))))] * invDt);
    float conv = as_float(0x00000000u);
    float diff = as_float(0x00000000u);
    uint begU = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    uint endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint fU = faceIdx[k];
      int f = ((int)(fU));
      if ((f >= counts3[1])) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = as_float(0x00000000u);
        float muF = as_float(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != (uint)(0u))) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != (uint)(0u))) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (as_float(0x00000000u) - rhoPhi);
        }
        if ((outFlux > as_float(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)((int)(get_global_id(0))))]);
    if ((A != as_float(0x00000000u))) {
      outRAU[((int)((int)(get_global_id(0))))] = (as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}
