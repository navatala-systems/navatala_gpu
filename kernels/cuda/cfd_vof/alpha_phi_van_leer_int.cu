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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_vof_alpha_phi_van_leer_int(const float* alpha, const float* phi, const int* owner, const int* nei, const float* gradX, const float* gradY, const float* gradZ, const float* cellCx, const float* cellCy, const float* cellCz, const float* weights, const float* phir, const float* alphaF, const int* faceParams, float* outAlphaPhi) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= faceParams[0]) {
    return;
  } else {
    float ph = phi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int o = owner[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    int n = nei[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float aO = alpha[o];
    float aN = alpha[n];
    float gradf = (aN - aO);
    float gxU = gradX[n];
    float gyU = gradY[n];
    float gzU = gradZ[n];
    if (ph > __uint_as_float(0x00000000u)) {
      gxU = gradX[o];
      gyU = gradY[o];
      gzU = gradZ[o];
    }
    float dx = (cellCx[n] - cellCx[o]);
    float dy = (cellCy[n] - cellCy[o]);
    float dz = (cellCz[n] - cellCz[o]);
    float gradcf = (((dx * gxU) + (dy * gyU)) + (dz * gzU));
    float absGradf = abs(gradf);
    float absGradcf = abs(gradcf);
    float gradfSign = __uint_as_float(0x3f800000u);
    if (gradf < __uint_as_float(0x00000000u)) {
      gradfSign = __uint_as_float(0xbf800000u);
    }
    float gradcfSign = __uint_as_float(0x3f800000u);
    if (gradcf < __uint_as_float(0x00000000u)) {
      gradcfSign = __uint_as_float(0xbf800000u);
    }
    float r = __uint_as_float(0x00000000u);
    if (absGradf > __uint_as_float(0x0da24260u)) {
      if (absGradcf >= (__uint_as_float(0x447a0000u) * absGradf)) {
        r = ((__uint_as_float(0x44fa0000u) * (gradcfSign * gradfSign)) - __uint_as_float(0x3f800000u));
      } else {
        r = ((__uint_as_float(0x40000000u) * (gradcf / gradf)) - __uint_as_float(0x3f800000u));
      }
    }
    float absR = abs(r);
    float psi = ((r + absR) / (__uint_as_float(0x3f800000u) + absR));
    if (((ph > __uint_as_float(0x00000000u)) && ((aO < __uint_as_float(0x00000000u)) || (aN > __uint_as_float(0x3f800000u)))) || ((ph < __uint_as_float(0x00000000u)) && ((aN < __uint_as_float(0x00000000u)) || (aO > __uint_as_float(0x3f800000u))))) {
      psi = __uint_as_float(0x00000000u);
    }
    float pos0 = __uint_as_float(0x00000000u);
    if (ph >= __uint_as_float(0x00000000u)) {
      pos0 = __uint_as_float(0x3f800000u);
    }
    float cdw = weights[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float w = ((psi * cdw) + ((__uint_as_float(0x3f800000u) - psi) * pos0));
    float oneMinusW = (__uint_as_float(0x3f800000u) - w);
    float a = ((w * aO) + (oneMinusW * aN));
    float ac = alphaF[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if (ac < __uint_as_float(0x00000000u)) {
      ac = __uint_as_float(0x00000000u);
    }
    if (ac > __uint_as_float(0x3f800000u)) {
      ac = __uint_as_float(0x3f800000u);
    }
    float oneMinusAc = (__uint_as_float(0x3f800000u) - ac);
    float comp = (ac * oneMinusAc);
    outAlphaPhi[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = ((ph * a) + (phir[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] * comp));
  }
}
