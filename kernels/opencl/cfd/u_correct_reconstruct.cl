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

__kernel void navatala_cfd_u_correct_reconstruct(__global const float* pCell, __global const float* rAU, __global const float* hbx, __global const float* hby, __global const float* hbz, __global const float* rAUf, __global const float* phig, __global const int* owner, __global const int* neighbour, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* magSf, __global const float* deltaCoeffs, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* bcVal, __global const int* bcMask, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = max(counts[0] - 1, 0);
  const int safeIdx = min(gid0, nSafeMax);
  if (gid0 >= counts[0]) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float a00 = as_float(0x00000000u);
    float a01 = as_float(0x00000000u);
    float a02 = as_float(0x00000000u);
    float a10 = as_float(0x00000000u);
    float a11 = as_float(0x00000000u);
    float a12 = as_float(0x00000000u);
    float a20 = as_float(0x00000000u);
    float a21 = as_float(0x00000000u);
    float a22 = as_float(0x00000000u);
    float bx = as_float(0x00000000u);
    float by = as_float(0x00000000u);
    float bz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if ((f < counts[1])) {
        int o = owner[f];
        float po = pCell[o];
        float other = po;
        if ((f < counts[2])) {
          int n = neighbour[f];
          other = pCell[n];
        } else {
          if ((bcMask[f] != 0)) {
            other = bcVal[f];
          }
        }
        float snGrad = (deltaCoeffs[f] * (other - po));
        float phiTerm = as_float(0x00000000u);
        if ((bcSnGradMask[f] == 2)) {
          phiTerm = bcSnGrad[f];
        } else {
          if ((bcSnGradMask[f] == 1)) {
            snGrad = bcSnGrad[f];
          }
          float gradFace = (magSf[f] * snGrad);
          float denom = rAUf[f];
          float pFlux = ((paramsF[0] * (denom * gradFace)) + faceFluxCorrection[f]);
          if ((denom != as_float(0x00000000u))) {
            phiTerm = ((phig[f] - pFlux) / denom);
          }
        }
        float ms = magSf[f];
        if ((ms > as_float(0x00000000u))) {
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
    float smallConst = as_float(0x33d6bf95u);
    float rootVSmall = as_float(0x1e3ce508u);
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
      w00 = (w00 + as_float(0x3f800000u));
    }
    if (small_yy) {
      w11 = (w11 + as_float(0x3f800000u));
    }
    if (small_zz) {
      w22 = (w22 + as_float(0x3f800000u));
    }
    float det = (((((w00 * w11) * w22) + ((w01 * w12) * w20)) + ((w02 * w10) * w21)) - ((((w00 * w12) * w21) + ((w01 * w10) * w22)) + ((w02 * w11) * w20)));
    float rx = as_float(0x00000000u);
    float ry = as_float(0x00000000u);
    float rz = as_float(0x00000000u);
    if ((fabs(det) >= rootVSmall)) {
      float invDet = (as_float(0x3f800000u) / det);
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
        i00 = (i00 - as_float(0x3f800000u));
      }
      if (small_yy) {
        i11 = (i11 - as_float(0x3f800000u));
      }
      if (small_zz) {
        i22 = (i22 - as_float(0x3f800000u));
      }
      rx = (((i00 * bx) + (i01 * by)) + (i02 * bz));
      ry = (((i10 * bx) + (i11 * by)) + (i12 * bz));
      rz = (((i20 * bx) + (i21 * by)) + (i22 * bz));
    }
    float fac = rAU[((int)((int)(get_global_id(0))))];
    outX[((int)((int)(get_global_id(0))))] = (hbx[((int)((int)(get_global_id(0))))] + (fac * rx));
    outY[((int)((int)(get_global_id(0))))] = (hby[((int)((int)(get_global_id(0))))] + (fac * ry));
    outZ[((int)((int)(get_global_id(0))))] = (hbz[((int)((int)(get_global_id(0))))] + (fac * rz));
  }
}
