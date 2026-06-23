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
extern "C" __global__ void navatala_cfd_k_omega_s_s_t_sources(const float* k, const float* omega, const float* nut, const float* nu, const float* alpha, const float* rho, const float* F1, const float* F23, const float* CDkOmega, const float* S2, const float* divU, const int* counts, float* outOmegaSp, float* outOmegaSuSp1, float* outOmegaSuSp2, float* outOmegaExpl, float* outKSp, float* outKSuSp, float* outKExpl, float* outDomegaEff, float* outDkEff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float omC = omega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nutC = nut[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float nuC = nu[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float alphaC = alpha[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float rhoC = rho[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float f1C = F1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float f23C = F23[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float cdkwC = CDkOmega[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float s2C = S2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float divUC = divU[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * __uint_as_float(0x3decbfb1u)) + __uint_as_float(0x3ee147aeu));
    float betaC = (__uint_as_float(0x3da9930cu) - (f1C * __uint_as_float(0x3bff9724u)));
    float alphaKC = (__uint_as_float(0x3f800000u) - (f1C * __uint_as_float(0x3e19999au)));
    float alphaOmC = (__uint_as_float(0x3f5b22d1u) - (f1C * __uint_as_float(0x3eb645a2u)));
    float s2Safe = (((s2C > __uint_as_float(0x00000000u))) ? (s2C) : (__uint_as_float(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (__uint_as_float(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (__uint_as_float(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (__uint_as_float(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (__uint_as_float(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (__uint_as_float(0x00000000u) - (__uint_as_float(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > __uint_as_float(0x26901d7du))) ? (omC) : (__uint_as_float(0x26901d7du)));
    float f1m1 = (f1C - __uint_as_float(0x3f800000u));
    float omSuSp2 = (__uint_as_float(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (__uint_as_float(0x00000000u) - (arC * (__uint_as_float(0x3db851ecu) * omC)));
    float kSuSp = (__uint_as_float(0x00000000u) - (__uint_as_float(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSp;
    outOmegaSuSp1[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSuSp1;
    outOmegaSuSp2[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omSuSp2;
    outOmegaExpl[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = omExpl;
    outKSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kSp;
    outKSuSp[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kSuSp;
    outKExpl[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = kExpl;
    outDomegaEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = domEff;
    outDkEff[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))] = dkEff;
  }
}
