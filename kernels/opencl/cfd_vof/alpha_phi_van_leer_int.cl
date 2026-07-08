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

__kernel void navatala_cfd_vof_alpha_phi_van_leer_int(__global const float* alpha, __global const float* phi, __global const int* owner, __global const int* nei, __global const float* gradX, __global const float* gradY, __global const float* gradZ, __global const float* cellCx, __global const float* cellCy, __global const float* cellCz, __global const float* weights, __global const float* phir, __global const float* alphaF, __global const int* faceParams, __global float* outAlphaPhi) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(get_global_id(0))))];
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float aO = alpha[o];
    float aN = alpha[n];
    float gradf = (aN - aO);
    float gxU = gradX[n];
    float gyU = gradY[n];
    float gzU = gradZ[n];
    if (ph > as_float(0x00000000u)) {
      gxU = gradX[o];
      gyU = gradY[o];
      gzU = gradZ[o];
    }
    float dx = (cellCx[n] - cellCx[o]);
    float dy = (cellCy[n] - cellCy[o]);
    float dz = (cellCz[n] - cellCz[o]);
    float gradcf = (((dx * gxU) + (dy * gyU)) + (dz * gzU));
    float absGradf = fabs(gradf);
    float absGradcf = fabs(gradcf);
    float gradfSign = as_float(0x3f800000u);
    if (gradf < as_float(0x00000000u)) {
      gradfSign = as_float(0xbf800000u);
    }
    float gradcfSign = as_float(0x3f800000u);
    if (gradcf < as_float(0x00000000u)) {
      gradcfSign = as_float(0xbf800000u);
    }
    float r = as_float(0x00000000u);
    if (absGradf > as_float(0x0da24260u)) {
      if (absGradcf >= (as_float(0x447a0000u) * absGradf)) {
        r = ((as_float(0x44fa0000u) * (gradcfSign * gradfSign)) - as_float(0x3f800000u));
      } else {
        r = ((as_float(0x40000000u) * (gradcf / gradf)) - as_float(0x3f800000u));
      }
    }
    float absR = fabs(r);
    float psi = ((r + absR) / (as_float(0x3f800000u) + absR));
    if (((ph > as_float(0x00000000u)) && ((aO < as_float(0x00000000u)) || (aN > as_float(0x3f800000u)))) || ((ph < as_float(0x00000000u)) && ((aN < as_float(0x00000000u)) || (aO > as_float(0x3f800000u))))) {
      psi = as_float(0x00000000u);
    }
    float pos0 = as_float(0x00000000u);
    if (ph >= as_float(0x00000000u)) {
      pos0 = as_float(0x3f800000u);
    }
    float cdw = weights[((int)((int)(get_global_id(0))))];
    float w = ((psi * cdw) + ((as_float(0x3f800000u) - psi) * pos0));
    float oneMinusW = (as_float(0x3f800000u) - w);
    float a = ((w * aO) + (oneMinusW * aN));
    float ac = alphaF[((int)((int)(get_global_id(0))))];
    if (ac < as_float(0x00000000u)) {
      ac = as_float(0x00000000u);
    }
    if (ac > as_float(0x3f800000u)) {
      ac = as_float(0x3f800000u);
    }
    float oneMinusAc = (as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinusAc);
    outAlphaPhi[((int)((int)(get_global_id(0))))] = ((ph * a) + (phir[((int)((int)(get_global_id(0))))] * comp));
  }
}
