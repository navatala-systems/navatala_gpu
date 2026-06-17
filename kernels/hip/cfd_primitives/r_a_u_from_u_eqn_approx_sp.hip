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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_cfd_primitives_r_a_u_from_u_eqn_approx_sp(const float* rhoCell, const float* muCell, const float* phiFaces, const float* rhoBcVal, const unsigned int* rhoBcMask, const float* muBcVal, const unsigned int* muBcMask, const int* ownerAll, const int* neighbourInt, const float* weightsAll, const float* magSfAll, const float* deltaAll, const unsigned int* offsets, const unsigned int* faceIdx, const int* sign, const float* volCells, const float* sp, const int* counts3, const float* paramsF, float* outRAU) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts3[0])) {
    return;
  } else {
    float invDt = (__uint_as_float(0x3f800000u) / paramsF[0]);
    float V = volCells[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float A = (rhoCell[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * invDt);
    float conv = __uint_as_float(0x00000000u);
    float diff = __uint_as_float(0x00000000u);
    unsigned int begU = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    unsigned int endU = offsets[c1];
    int beg = ((int)(begU));
    int end = ((int)(endU));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      unsigned int fU = faceIdx[k];
      int f = ((int)(fU));
      if ((f >= counts3[1])) {
      } else {
        int s = sign[k];
        int o = ownerAll[f];
        float phi = phiFaces[f];
        float rhoF = __uint_as_float(0x00000000u);
        float muF = __uint_as_float(0x00000000u);
        if ((f < counts3[2])) {
          int n = neighbourInt[f];
          float w = weightsAll[f];
          float iw = (__uint_as_float(0x3f800000u) - w);
          rhoF = ((w * rhoCell[o]) + (iw * rhoCell[n]));
          muF = ((w * muCell[o]) + (iw * muCell[n]));
        } else {
          if ((rhoBcMask[f] != 0u)) {
            rhoF = rhoBcVal[f];
          } else {
            rhoF = rhoCell[o];
          }
          if ((muBcMask[f] != 0u)) {
            muF = muBcVal[f];
          } else {
            muF = muCell[o];
          }
        }
        float rhoPhi = (rhoF * phi);
        float outFlux = rhoPhi;
        if ((s < 0)) {
          outFlux = (__uint_as_float(0x00000000u) - rhoPhi);
        }
        if ((outFlux > __uint_as_float(0x00000000u))) {
          conv = (conv + outFlux);
        }
        float diffTerm = (muF * (magSfAll[f] * deltaAll[f]));
        diff = (diff + diffTerm);
      }
    }
    A = (A + ((conv + diff) / V));
    A = (A + sp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))]);
    if ((A != __uint_as_float(0x00000000u))) {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (__uint_as_float(0x3f800000u) / A);
    } else {
      outRAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = __uint_as_float(0x00000000u);
    }
  }
}
