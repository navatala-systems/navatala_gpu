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
extern "C" __global__ void navatala_cfd_u_correct_reconstruct(const float* pCell, const float* rAU, const float* hbx, const float* hby, const float* hbz, const float* rAUf, const float* phig, const int* owner, const int* neighbour, const float* sfX, const float* sfY, const float* sfZ, const float* magSf, const float* deltaCoeffs, const int* offsets, const int* faceIdx, const float* sign, const float* vol, const float* bcVal, const int* bcMask, const float* bcSnGrad, const int* bcSnGradMask, const float* faceFluxCorrection, const int* counts, const float* paramsF, float* outX, float* outY, float* outZ) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int c1 = (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float a00 = __uint_as_float(0x00000000u);
    float a01 = __uint_as_float(0x00000000u);
    float a02 = __uint_as_float(0x00000000u);
    float a10 = __uint_as_float(0x00000000u);
    float a11 = __uint_as_float(0x00000000u);
    float a12 = __uint_as_float(0x00000000u);
    float a20 = __uint_as_float(0x00000000u);
    float a21 = __uint_as_float(0x00000000u);
    float a22 = __uint_as_float(0x00000000u);
    float bx = __uint_as_float(0x00000000u);
    float by = __uint_as_float(0x00000000u);
    float bz = __uint_as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int o = owner[f];
        float po = pCell[o];
        float other = po;
        if (f < counts[2]) {
          int n = neighbour[f];
          other = pCell[n];
        } else {
          if (bcMask[f] != 0) {
            other = bcVal[f];
          }
        }
        float snGrad = (deltaCoeffs[f] * (other - po));
        float phiTerm = __uint_as_float(0x00000000u);
        if (bcSnGradMask[f] == 2) {
          phiTerm = bcSnGrad[f];
        } else {
          if (bcSnGradMask[f] == 1) {
            snGrad = bcSnGrad[f];
          }
          float gradFace = (magSf[f] * snGrad);
          float denom = rAUf[f];
          float pFlux = ((paramsF[0] * (denom * gradFace)) + faceFluxCorrection[f]);
          if (denom != __uint_as_float(0x00000000u)) {
            phiTerm = ((phig[f] - pFlux) / denom);
          }
        }
        float ms = magSf[f];
        if (ms > __uint_as_float(0x00000000u)) {
          float nx = (sfX[f] / ms);
          float ny = (sfY[f] / ms);
          float nz = (sfZ[f] / ms);
          a00 = (a00 + (nx * sfX[f]));
          a01 = (a01 + (nx * sfY[f]));
          a02 = (a02 + (nx * sfZ[f]));
          a10 = (a10 + (ny * sfX[f]));
          a11 = (a11 + (ny * sfY[f]));
          a12 = (a12 + (ny * sfZ[f]));
          a20 = (a20 + (nz * sfX[f]));
          a21 = (a21 + (nz * sfY[f]));
          a22 = (a22 + (nz * sfZ[f]));
          bx = (bx + (nx * phiTerm));
          by = (by + (ny * phiTerm));
          bz = (bz + (nz * phiTerm));
        }
      }
    }
    float smallConst = __uint_as_float(0x33d6bf95u);
    float rootVSmall = __uint_as_float(0x1e3ce508u);
    float magSqr_xx = (a00 * a00);
    float magSqr_yy = (a11 * a11);
    float magSqr_zz = (a22 * a22);
    float threshold = (smallConst * ((magSqr_xx + magSqr_yy) + magSqr_zz));
    bool small_xx = (magSqr_xx < threshold);
    bool small_yy = (magSqr_yy < threshold);
    bool small_zz = (magSqr_zz < threshold);
    float w00 = a00;
    float w01 = a01;
    float w02 = a02;
    float w10 = a10;
    float w11 = a11;
    float w12 = a12;
    float w20 = a20;
    float w21 = a21;
    float w22 = a22;
    if (small_xx) {
      w00 = (w00 + __uint_as_float(0x3f800000u));
    }
    if (small_yy) {
      w11 = (w11 + __uint_as_float(0x3f800000u));
    }
    if (small_zz) {
      w22 = (w22 + __uint_as_float(0x3f800000u));
    }
    float det = (((((w00 * w11) * w22) + ((w01 * w12) * w20)) + ((w02 * w10) * w21)) - ((((w00 * w12) * w21) + ((w01 * w10) * w22)) + ((w02 * w11) * w20)));
    float rx = __uint_as_float(0x00000000u);
    float ry = __uint_as_float(0x00000000u);
    float rz = __uint_as_float(0x00000000u);
    if (abs(det) >= rootVSmall) {
      float invDet = (__uint_as_float(0x3f800000u) / det);
      float i00 = (((w11 * w22) - (w21 * w12)) * invDet);
      float i01 = (((w02 * w21) - (w01 * w22)) * invDet);
      float i02 = (((w01 * w12) - (w02 * w11)) * invDet);
      float i10 = (((w12 * w20) - (w10 * w22)) * invDet);
      float i11 = (((w00 * w22) - (w02 * w20)) * invDet);
      float i12 = (((w02 * w10) - (w00 * w12)) * invDet);
      float i20 = (((w10 * w21) - (w11 * w20)) * invDet);
      float i21 = (((w01 * w20) - (w00 * w21)) * invDet);
      float i22 = (((w00 * w11) - (w10 * w01)) * invDet);
      if (small_xx) {
        i00 = (i00 - __uint_as_float(0x3f800000u));
      }
      if (small_yy) {
        i11 = (i11 - __uint_as_float(0x3f800000u));
      }
      if (small_zz) {
        i22 = (i22 - __uint_as_float(0x3f800000u));
      }
      rx = (((i00 * bx) + (i01 * by)) + (i02 * bz));
      ry = (((i10 * bx) + (i11 * by)) + (i12 * bz));
      rz = (((i20 * bx) + (i21 * by)) + (i22 * bz));
    }
    float fac = rAU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    outX[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hbx[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * rx));
    outY[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hby[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * ry));
    outZ[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = (hbz[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] + (fac * rz));
  }
}
