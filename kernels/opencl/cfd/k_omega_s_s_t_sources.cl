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

__kernel void navatala_cfd_k_omega_s_s_t_sources(__global const float* k, __global const float* omega, __global const float* nut, __global const float* nu, __global const float* alpha, __global const float* rho, __global const float* F1, __global const float* F23, __global const float* CDkOmega, __global const float* S2, __global const float* divU, __global const int* counts, __global float* outOmegaSp, __global float* outOmegaSuSp1, __global float* outOmegaSuSp2, __global float* outOmegaExpl, __global float* outKSp, __global float* outKSuSp, __global float* outKExpl, __global float* outDomegaEff, __global float* outDkEff) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((((int)((int)(get_global_id(0)))) >= counts[0])) {
    return;
  } else {
    float kC = k[((int)((int)(get_global_id(0))))];
    float omC = omega[((int)((int)(get_global_id(0))))];
    float nutC = nut[((int)((int)(get_global_id(0))))];
    float nuC = nu[((int)((int)(get_global_id(0))))];
    float alphaC = alpha[((int)((int)(get_global_id(0))))];
    float rhoC = rho[((int)((int)(get_global_id(0))))];
    float f1C = F1[((int)((int)(get_global_id(0))))];
    float f23C = F23[((int)((int)(get_global_id(0))))];
    float cdkwC = CDkOmega[((int)((int)(get_global_id(0))))];
    float s2C = S2[((int)((int)(get_global_id(0))))];
    float divUC = divU[((int)((int)(get_global_id(0))))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * as_float(0x3decbfb1u)) + as_float(0x3ee147aeu));
    float betaC = (as_float(0x3da9930cu) - (f1C * as_float(0x3bff9724u)));
    float alphaKC = (as_float(0x3f800000u) - (f1C * as_float(0x3e19999au)));
    float alphaOmC = (as_float(0x3f5b22d1u) - (f1C * as_float(0x3eb645a2u)));
    float s2Safe = (((s2C > as_float(0x00000000u))) ? (s2C) : (as_float(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (as_float(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (as_float(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (as_float(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (as_float(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (as_float(0x00000000u) - (as_float(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > as_float(0x26901d7du))) ? (omC) : (as_float(0x26901d7du)));
    float f1m1 = (f1C - as_float(0x3f800000u));
    float omSuSp2 = (as_float(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (as_float(0x00000000u) - (arC * (as_float(0x3db851ecu) * omC)));
    float kSuSp = (as_float(0x00000000u) - (as_float(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)((int)(get_global_id(0))))] = omSp;
    outOmegaSuSp1[((int)((int)(get_global_id(0))))] = omSuSp1;
    outOmegaSuSp2[((int)((int)(get_global_id(0))))] = omSuSp2;
    outOmegaExpl[((int)((int)(get_global_id(0))))] = omExpl;
    outKSp[((int)((int)(get_global_id(0))))] = kSp;
    outKSuSp[((int)((int)(get_global_id(0))))] = kSuSp;
    outKExpl[((int)((int)(get_global_id(0))))] = kExpl;
    outDomegaEff[((int)((int)(get_global_id(0))))] = domEff;
    outDkEff[((int)((int)(get_global_id(0))))] = dkEff;
  }
}
